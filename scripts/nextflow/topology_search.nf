params.root_dir       = '/Users/schmidt73/Desktop/fastLAML/'
params.sim_dir        = "${params.root_dir}/examples/"
params.fast_laml      = "${params.root_dir}/build/src/fastlaml"
params.outdir         = "${params.root_dir}/nextflow_results"

params.generation_script = "${params.root_dir}/scripts/generate_starting_trees.R"
params.R_bin = 'Rscript'

params.instances = [
    [id: 'k400_s134_sub100_r01', gaussian_overlap: 0.01, dropout_rate: 0.1]
]

process generate_starting_trees {
    memory '2 GB'
    time '59m'
    stageInMode 'copy'

    publishDir "${params.outdir}/starting_trees/${id}", mode: 'copy'

    input:
        tuple val(args), val(id), path(character_matrix)

    output:
        tuple val(args), path("initial.*.nwk")

    """
    ${params.R_bin} ${params.generation_script} --input ${character_matrix} --nrep 3 --output initial --seed 0
    """
}

process fast_laml {
    memory '2 GB'
    time '59m'
    cpus 1
    stageInMode 'copy'

    publishDir "${params.outdir}/fast-laml/${id}/${starting_tree_id}/", mode: 'copy'

    input:
        tuple val(id), val(starting_tree_id), path(matrix), val(matrix_format), path(tree)

    output:
        tuple path(tree), 
            path("fastlaml_ultrametric_tree.nwk"), 
            path("fastlaml_tree.nwk"), 
            path("fastlaml_results.json"),
            path("timing.txt")

    """
    gtime -v ${params.fast_laml} --matrix ${matrix} --tree ${tree} \
                        --output fastlaml --mode search --seed 0 \
                        --temp 0.000001 --max-iterations 1000 -d ${matrix_format} &> timing.txt
    mv fastlaml_ultrametric_tree.newick fastlaml_ultrametric_tree.nwk
    mv fastlaml_tree.newick fastlaml_tree.nwk 
    """
}

workflow {
    parameter_channel = channel.fromList(params.instances)

    // pulls simulation files from specified directories
    simulations = parameter_channel | map { inst ->
        id = inst.id
        gaussian_overlap = inst.gaussian_overlap
        dropout_rate = inst.dropout_rate

        dir_prefix              = "${params.sim_dir}/${id}/"
        character_matrix        = file("${params.sim_dir}/${id}/character_matrix.csv")
        tree                    = file("${params.sim_dir}/${id}/tree.nwk")
        argmax_character_matrix = file("${params.sim_dir}/${id}/${id}_r${gaussian_overlap}_p${dropout_rate}_argmax.csv")
        observation_matrix      = file("${params.sim_dir}/${id}/${id}_r${gaussian_overlap}_p${dropout_rate}_scores.format.csv")
        ["${id}_r${gaussian_overlap}_p${dropout_rate}", tree, character_matrix, argmax_character_matrix, observation_matrix]
    }

    // takes the "cross product" of the simulations and the starting trees
    initialized_simulations = simulations | map { args ->
        [args, args[1], args[3]]
    } | generate_starting_trees | flatMap { args, trees ->
        res = []
        for (tree in trees) {
            item = args.clone()
            item.add(tree)
            res.add(item)
        }
        res
    }

    // runs fast-laml simulations for the argmax character matrix
    initialized_simulations | map { 
        id, true_tree, character_matrix, argmax_character_matrix, observation_matrix, starting_tree ->
        matrix = argmax_character_matrix
        matrix_format = 'character-matrix'
        [id, starting_tree.getName(), matrix, matrix_format, starting_tree]
    } | fast_laml
}
