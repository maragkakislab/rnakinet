rule generate_pod5_metrics:
    input:
        experiment_dir = lambda wildcards: f"{DATA_DIR}/{EXP_TO_PATH[wildcards.experiment_name]}",
    output:
        csv_path = OUTPUTS_DIR + "/diagnostics/pod5_metrics/{experiment_name}_sr_{sampling_rate}_metrics.csv"
    conda:
        "../envs/pod5_splitting.yaml"
    shell:
        """
        python scripts/pod5_metrics.py \
            --experiment-dir {input.experiment_dir} \
            --output {output.csv_path} \
            --sampling-rate {wildcards.sampling_rate}
        """

rule plot_pod5_metrics:
    input:
        input_csvs=lambda wildcards: expand(
            OUTPUTS_DIR + "/diagnostics/pod5_metrics/{experiment_name}_sr_{sampling_rate}_metrics.csv",
            experiment_name=DIAGNOSTICS_GRAPHING_PARAMS[wildcards.plot_name]["experiments"],
            sampling_rate=DIAGNOSTICS_GRAPHING_PARAMS[wildcards.plot_name]["sampling_rate"],
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
        python scripts/plot_pod5_metrics.py \
            --title "{params.title}" \
            --metric "{params.metric}" \
            --input-csvs {input.input_csvs} \
            --output "{output.plot}" \
            --normalize
        """