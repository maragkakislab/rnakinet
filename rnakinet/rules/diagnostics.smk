rule generate_pod5_stats:
    input:
        experiment_dir = lambda wildcards: f"{DATA_DIR}/{EXP_TO_PATH[wildcards.experiment_name]}",
    output:
        csv_path = OUTPUTS_DIR + "/diagnostics/pod5_stats/{experiment_name}_sr_{subsample}_stats.csv"
    conda:
        "../envs/pod5_splitting.yaml"
    shell:
        """
        python scripts/pod5_stats.py \
            --experiment-dir {input.experiment_dir} \
            --output {output.csv_path} \
            --subsample {wildcards.subsample}
        """

rule plot_pod5_stats:
    input:
        input_csvs=lambda wildcards: expand(
            OUTPUTS_DIR + "/diagnostics/pod5_stats/{experiment_name}_sr_{subsample}_stats.csv",
            experiment_name=DIAGNOSTICS_GRAPHING_PARAMS[wildcards.plot_name]["experiments"],
            subsample=DIAGNOSTICS_GRAPHING_PARAMS[wildcards.plot_name]["subsample"],
        )
    output:
        plot=OUTPUTS_DIR + "/diagnostics/plots/{plot_name}.png"
    conda:
        "../envs/visual.yaml"
    params:
        title=lambda wildcards: wildcards.plot_name,
        metric=lambda wildcards: DIAGNOSTICS_GRAPHING_PARAMS[wildcards.plot_name]["metric"],
    shell:
        ## TODO use the normalize param from config and add log10 option to config as well
        """
        python scripts/plot_pod5_stats.py \
            --title "{params.title}" \
            --metric "{params.metric}" \
            --input-csvs {input.input_csvs} \
            --output "{output.plot}" \
            --normalize
        """