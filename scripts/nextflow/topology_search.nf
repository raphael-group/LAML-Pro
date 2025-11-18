params.root_dir       = '/n/fs/ragr-research/projects/fast-laml/'
params.sim_dir        = "/n/fs/ragr-research/projects/laml-pro/sim_data/set_3d/input/"
params.fast_laml      = "${params.root_dir}/build/src/fastlaml"
params.outdir         = "${params.root_dir}/nextflow_results"

params.convexml = "${params.root_dir}/scripts/run_convexml.py"
params.nj_script = "${params.root_dir}/scripts/neighbor_joining.py"
params.generation_script = "${params.root_dir}/scripts/generate_starting_trees.R"
params.R_bin = '/n/fs/ragr-data/users/schmidt/miniconda3/envs/breaked/bin/Rscript'

def create_instances(cells, replicate) {
  return [
    [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.0439', dropout_rate: '0.0'],
    [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.249', dropout_rate: '0.0'],
    [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.458', dropout_rate: '0.0'],
    [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.607', dropout_rate: '0.0'],
    [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.0439', dropout_rate: '0.1875'],
    [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.249', dropout_rate: '0.1875'],
    [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.458', dropout_rate: '0.1875'],
    [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.607', dropout_rate: '0.1875']
    // [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.135', dropout_rate: '0.0'],
    // [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.36', dropout_rate: '0.0'],
    // [id: "k400_s0_sub${cells}_r${replicate}",  gaussian_overlap: '0.539', dropout_rate: '0.0'],
    // [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.0', dropout_rate: '0.1875'],
    // [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.135', dropout_rate: '0.1875'],
    // [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.36', dropout_rate: '0.1875'],
    // [id: "k400_s15_sub${cells}_r${replicate}", gaussian_overlap: '0.539', dropout_rate: '0.1875'],
  ]
}

params.instances = [
    // create_instances("100", "01"),
    // create_instances("100", "02"),
    create_instances("100", "03"),
    create_instances("100", "04"),
    create_instances("100", "05"),
    // create_instances("200", "01"),
    // create_instances("200", "02"),
    create_instances("200", "03"),
    create_instances("200", "04"),
    create_instances("200", "05"),
    // create_instances("300", "01"),
    // create_instances("300", "02"),
    create_instances("300", "03"),
    create_instances("300", "04"),
    create_instances("300", "05")
].collectMany { it }

process generate_starting_trees {
    memory '2 GB'
    time '59m'
    cpus 1

    stageInMode 'copy'
    publishDir "${params.outdir}/starting_trees/${id}", mode: 'copy'

    input:
        tuple val(args), val(id), path(missing_character_matrix), path(argmax_character_matrix)

    output:
        tuple val(args), path("initial.*.nwk")

    """
    ${params.R_bin} ${params.generation_script} --input ${missing_character_matrix} --nrep 4 --output initial.missing --seed 0
    ${params.R_bin} ${params.generation_script} --input ${argmax_character_matrix} --nrep 4 --output initial.argmax --seed 0
    """
}

process fast_laml {
    memory '2 GB'
    time '4h'
    cpus 1

    errorStrategy 'ignore'
    clusterOptions '--partition=preempt'
    stageInMode 'copy'
    publishDir "${params.outdir}/fast-laml/${id}/${starting_tree_id}/", mode: 'copy'

    input:
        tuple val(id), val(starting_tree_id), path(matrix), val(matrix_format), path(tree)

    output:
        tuple path(tree), 
              path("fastlaml_tree.nwk"), 
              path("fastlaml_results.json"),
              //path("fastlaml_posterior_probs.csv"),
              //path("fastlaml_posterior_argmax.csv"),
              path("timing.txt")

    """
    /usr/bin/time -v ${params.fast_laml} --matrix ${matrix} --tree ${tree} \
                        --output fastlaml --mode search --seed 0 --ultrametric \
                        --temp 0.01 --max-iterations 5000 -d ${matrix_format} &> timing.txt
    mv fastlaml_tree.newick fastlaml_tree.nwk 
    """
}

process convexml {
    memory '2 GB'
    time '59m'
    cpus 1

    errorStrategy 'ignore'
    clusterOptions '--partition=preempt'
    stageInMode 'copy'
    publishDir "${params.outdir}/convex-ml/${id}/", mode: 'copy'

    input:
        tuple val(id), path(matrix), val(tree)

    output:
        tuple path("tree.nwk"), path("timing.txt")

    script:
        """
        /usr/bin/time -v ${params.convexml} ${matrix} ${tree} tree.nwk &> timing.txt
        """
}

process neighbor_joining {
    memory '2 GB'
    time '59m'
    cpus 1

    errorStrategy 'ignore'
    clusterOptions '--partition=preempt'
    stageInMode 'copy'
    publishDir "${params.outdir}/neighbor-joining-${distance}/${id}/", mode: 'copy'

    input:
        tuple val(id), path(matrix), val(distance)

    output:
        tuple val("${id}/neighbor-joining-${distance}"), path(matrix), path("tree.nwk"), path("timing.txt")

    script:
    if (distance == "hamming")
        """
        /usr/bin/time -v python ${params.nj_script} ${matrix} result &> timing.txt
        mv result_hamming_tree.nwk tree.nwk
        """
    else if (distance == "weighted-hamming")
        """
        /usr/bin/time -v python ${params.nj_script} ${matrix} result &> timing.txt
        mv result_weighted_hamming_tree.nwk tree.nwk
        """
    else
        error "Invalid distance function: ${distance}"
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
        missing_character_matrix = file("${params.sim_dir}/${id}/${id}_dim3_r${gaussian_overlap}_p${dropout_rate}_only_missing.csv")
        argmax_character_matrix = file("${params.sim_dir}/${id}/${id}_dim3_r${gaussian_overlap}_p${dropout_rate}_argmax.csv")
        observation_matrix      = file("${params.sim_dir}/${id}/${id}_dim3_r${gaussian_overlap}_p${dropout_rate}_scores.csv")
        ["${id}_r${gaussian_overlap}_p${dropout_rate}", tree, missing_character_matrix, argmax_character_matrix, observation_matrix]
    }

    /* Run fastLAML */

    // takes the "cross product" of the simulations and the starting trees
    initialized_simulations = simulations | map { args ->
        [args, args[0], args[2], args[3]] 
    } | generate_starting_trees | flatMap { args, trees ->
        res = []
        for (tree in trees) {
            item = args.clone()
            item.add(tree)
            res.add(item)
        }
        res
    }

    // runs fast-laml simulations in three different settings
    // settings = channel.fromList(['observation-matrix', 'character-matrix', 'argmax-character-matrix'])
    settings = channel.fromList(['character-matrix', 'argmax-character-matrix']) //, 'character-matrix', 'argmax-character-matrix'])

    initialized_simulations | combine(settings) | map {
        id, true_tree, character_matrix, argmax_character_matrix, observation_matrix, starting_tree, setting ->

        has_argmax_starting_tree = starting_tree.toString().contains("argmax")

        if (setting == 'observation-matrix' && has_argmax_starting_tree) {
            matrix = observation_matrix
            matrix_format = 'observation-matrix'
        } else if (setting == 'character-matrix' && !has_argmax_starting_tree) {
            matrix = character_matrix
            matrix_format = 'character-matrix'
        } else if (setting == 'argmax-character-matrix' && has_argmax_starting_tree) {
            matrix = argmax_character_matrix
            matrix_format = 'character-matrix'
        } else {
            return null
        }

        return ["${id}.${setting}", starting_tree.getName(), matrix, matrix_format, starting_tree]
    } | fast_laml

    /* Run Neighbor Joining and ConvexML */

    // distances = channel.fromList(['hamming', 'weighted-hamming'])
    // simulations | map { args -> [args[0], args[3]] } | combine(distances) | neighbor_joining | map { args -> [args[0], args[1], args[2]] } | convexml
}
