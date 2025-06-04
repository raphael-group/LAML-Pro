params.root_dir       = '/Users/schmidt73/Desktop/fastLAML/'
params.sim_dir        = "${params.root_dir}/examples/"
params.fast_laml      = "${params.root_dir}/build/src/fastlaml"
params.outdir         = "${params.root_dir}/nextflow_results"

params.instances = [
    [id: 'k400_s134_sub100_r01', gaussian_overlap: 0.01, dropout_rate: 0.1]
]

process fast_laml {
    memory '2 GB'
    time '59m'
    stageInMode 'copy'

    publishDir "${params.outdir}/fast-laml/${id}", mode: 'copy'

    input:
      tuple val(id), path(matrix), val(matrix_format), path(tree)

    output:
      tuple path(tree), path("fastlaml_ultrametric_tree.nwk"), path("fastlaml_tree.nwk"), path("fastlaml_results.json")

    """
    ${params.fast_laml} --matrix ${matrix} --tree ${tree} \
                        --output fast_laml --mode search --seed 0 \
                        --temp 0.000001 --max-iterations 5000 -d ${matrix_format} 
    mv fastlaml_ultrametric_tree.newick fastlaml_ultrametric_tree.nwk
    mv fastlaml_tree.newick fastlaml_tree.nwk 
    """
}

workflow {
    parameter_channel = channel.fromList(params.instances)

    simulations = parameter_channel | map { inst ->
        id = inst.id
        gaussian_overlap = inst.gaussian_overlap
        dropout_rate = inst.dropout_rate

        dir_prefix              = "${params.sim_dir}/${id}/"
        character_matrix        = file("${params.sim_dir}/${id}/character_matrix.csv")
        tree                    = file("${params.sim_dir}/${id}/tree.nwk")
        argmax_character_matrix = file("${params.sim_dir}/${id}/${id}_r${gaussian_overlap}_p${dropout_rate}_argmax.csv")
        observation_matrix      = file("${params.sim_dir}/${id}/${id}_r${gaussian_overlap}_p${dropout_rate}_scores.format.csv")
        [id, tree, character_matrix, argmax_character_matrix, observation_matrix]
    }

    simulations | map { id, tree, character_matrix, argmax_character_matrix, observation_matrix ->
        matrix = argmax_character_matrix
        matrix_format = 'character-matrix'
        [id, matrix, matrix_format, tree]
    } | fast_laml
}
