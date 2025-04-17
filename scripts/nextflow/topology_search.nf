params.root_dir       = '/Users/schmidt73/Desktop/fastLAML/build'
params.sim_dir        = "${params.root_dir}/data/sims"
params.fast_laml      = "${params.root_dir}/build/src/laml"
params.outdir         = "${params.root_dir}/nextflow_results"

params.ncells        = [250]
params.dropout       = [30]
params.silencing     = [50]
params.seq_seed      = ["01"]
params.prior_seed    = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

process fast_laml {
    memory '4 GB'
    time '10m'
    // clusterOptions '--account=raphael --gres=gpu:1'
    // clusterOptions '--gres=gpu:1'

    publishDir "${params.outdir}/fast-laml-${mode}/${id}", mode: 'copy'

    input:
      tuple val(id), path(character_matrix), path(tree)

    output:
      tuple path("fastlaml_ultrametric_tree.newick"), path("fastlaml_tree.newick"), path("fastlaml_results.json")

    """
    ${params.fast_laml} --character-matrix ${character_matrix} --tree ${tree} --output fast_laml --mode search --seed 0 \
                        --temp 0.001 --max-iterations 500
    """
}

workflow {
    parameter_channel = channel.fromList(params.ncells)
                               .combine(channel.fromList(params.dropout))
                               .combine(channel.fromList(params.silencing))
                               .combine(channel.fromList(params.seq_seed))
                               .combine(channel.fromList(params.prior_seed))

    simulations = parameter_channel | map { ncells, dropout, silencing, seq_seed, prior_seed ->
        id = "s${silencing}d${dropout}p${prior_seed}_sub${ncells}_r${seq_seed}"
        character_matrix = "${params.sim_dir}/${id}/character_matrix.csv"
        startle_tree     = "${params.sim_dir}/${id}/startlenni_tree.nwk"
        tree             = "${params.sim_dir}/${id}/tree.nwk"
        [id, character_matrix, startle_tree]
    }

    simulations | view
}
