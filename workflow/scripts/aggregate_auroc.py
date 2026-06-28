import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pysam


def load_read_to_transcript(transcriptome_bam):
    with pysam.AlignmentFile(transcriptome_bam, "rb") as bamfile:
        rows = [
            {"read_id": read.query_name, "transcript": read.reference_name}
            for read in bamfile
            if not read.is_unmapped and not read.is_supplementary
        ]

    return pd.DataFrame(rows)


def load_predictions(prediction_paths):
    frames = []
    for prediction_path in prediction_paths:
        preds_df = pd.read_csv(prediction_path)
        if "score" not in preds_df.columns:
            preds_df = preds_df.rename(columns={"5eu_mod_score": "score"})

        missing_columns = {"read_id", "score"} - set(preds_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{prediction_path} is missing required columns: {missing}")

        frames.append(preds_df[["read_id", "score"]])

    if not frames:
        raise ValueError("At least one prediction file is required")

    read_to_score = pd.concat(frames, ignore_index=True)
    return read_to_score.dropna(subset=["read_id", "score"])


def attach_reference_ids(transcriptome_bam, prediction_paths, transcript_to_gene):
    read_to_transcript = load_read_to_transcript(transcriptome_bam)
    read_to_reference = read_to_transcript.merge(
        transcript_to_gene,
        left_on="transcript",
        right_on="Transcript stable ID version",
        how="left",
    )

    read_to_score = load_predictions(prediction_paths)
    read_reference_score = read_to_score.merge(read_to_reference, how="left", on="read_id")

    discarded = read_reference_score["Gene stable ID"].isnull().sum()
    total = len(read_reference_score)
    print(f"{discarded} / {total} reads not paired with a gene - discarded")

    return read_reference_score.dropna(subset=["Gene stable ID"])


def auroc_from_ranks(labels, scores):
    labels = np.asarray(labels)
    positive = labels == 1
    num_positive = positive.sum()
    num_negative = len(labels) - num_positive
    if num_positive == 0 or num_negative == 0:
        return np.nan

    ranks = pd.Series(scores).rank(method="average").to_numpy()
    positive_rank_sum = ranks[positive].sum()
    return (
        positive_rank_sum - (num_positive * (num_positive + 1) / 2)
    ) / (num_positive * num_negative)


def reference_stats(positive_reads, negative_reads, reference_column, threshold):
    combined = pd.concat(
        [
            positive_reads.assign(label=1),
            negative_reads.assign(label=0),
        ],
        ignore_index=True,
    )

    results = []
    for reference_id, group in combined.groupby(reference_column):
        if group["label"].nunique() < 2:
            continue

        positive_scores = group.loc[group["label"] == 1, "score"]
        negative_scores = group.loc[group["label"] == 0, "score"]
        results.append(
            {
                reference_column: reference_id,
                "auroc": auroc_from_ranks(group["label"], group["score"]),
                "pct_above_threshold_positive": (positive_scores > threshold).mean(),
                "pct_above_threshold_negative": (negative_scores > threshold).mean(),
                "num_positive_reads": len(positive_scores),
                "num_negative_reads": len(negative_scores),
            }
        )

    return pd.DataFrame(results)


def write_table(stats_df, output_path, level_name):
    if stats_df.empty:
        raise ValueError(f"Empty {level_name} level AUROC table")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, sep="\t", index=False)


def main(args):
    transcript_to_gene = pd.read_csv(args.transcript_to_gene_table, sep="\t")

    positive_reads = attach_reference_ids(
        args.positive_transcriptome_bam,
        args.positive_predictions,
        transcript_to_gene,
    )
    negative_reads = attach_reference_ids(
        args.negative_transcriptome_bam,
        args.negative_predictions,
        transcript_to_gene,
    )

    transcript_results = reference_stats(
        positive_reads,
        negative_reads,
        "Transcript stable ID",
        args.threshold,
    )
    gene_results = reference_stats(
        positive_reads,
        negative_reads,
        "Gene stable ID",
        args.threshold,
    )

    write_table(transcript_results, args.output_transcript_aurocs, "transcript")
    write_table(gene_results, args.output_gene_aurocs, "gene")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create gene- and transcript-level AUROC tables from read predictions"
    )
    parser.add_argument(
        "--positive-predictions",
        nargs="+",
        required=True,
        help="Positive-class read prediction CSV file(s)",
    )
    parser.add_argument(
        "--negative-predictions",
        nargs="+",
        required=True,
        help="Negative-class read prediction CSV file(s)",
    )
    parser.add_argument(
        "--positive-transcriptome-bam",
        required=True,
        help="Transcriptome BAM for positive-class reads",
    )
    parser.add_argument(
        "--negative-transcriptome-bam",
        required=True,
        help="Transcriptome BAM for negative-class reads",
    )
    parser.add_argument(
        "--transcript-to-gene-table",
        required=True,
        help="Table mapping transcript IDs to gene IDs",
    )
    parser.add_argument(
        "--output-gene-aurocs",
        required=True,
        help="Output TSV for gene-level AUROCs",
    )
    parser.add_argument(
        "--output-transcript-aurocs",
        required=True,
        help="Output TSV for transcript-level AUROCs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for positive/negative threshold fractions",
    )

    main(parser.parse_args())
