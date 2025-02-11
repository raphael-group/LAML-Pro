params.simulation_dir = '/n/fs/ragr-research/projects/problin_experiments/sim_tlscl_r2'
params.outdir         = 'nextflow_results'   
params.laml           = '/n/fs/ragr-research/projects/LAML/run_laml.py'
params.fast_laml      = '/n/fs/ragr-research/projects/fast-laml/src/laml.py'

params.ncells        = [250, 500, 1000, 10000]
params.ncharacters   = [30]
params.alphabet_size = [30, 100]
params.seq_seed      = ["01", "02", "03", "04", "05"]
params.prior_seed    = ["01", "02"]

process fast_laml {
    memory '4 GB'
    time '10m'
    clusterOptions '--account=raphael --gres=gpu:1'

    publishDir "${params.outdir}/fast-laml/${id}", mode: 'copy'

    input:
      tuple val(id), path(character_matrix), path(tree)

    output:
      tuple path("log.txt"), path("fast_laml_results.json")

    """
    python ${params.fast_laml} -t ${tree} -c ${character_matrix} --nu 0.5 --phi 0.5 --mode optimize -o fast_laml 2> log.txt
    """
}

process laml {
    cpus 8
    memory '4 GB'
    time '10m'
    clusterOptions '--account=raphael --gres=gpu:1'

    publishDir "${params.outdir}/laml/${id}", mode: 'copy'

    input:
      tuple val(id), path(character_matrix), path(tree)

    output:
      tuple path("LAML_output_params.txt"), path("LAML_output_trees.nwk"), path("LAML_output.log")

    """
    export MOSEKLM_LICENSE_FILE=/n/fs/grad/hs2435
    python ${params.laml} -c ${character_matrix} -t ${tree} --nInitials 1
    """
}

workflow {
    parameter_channel = channel.fromList(params.ncells)
                               .combine(channel.fromList(params.ncharacters))
                               .combine(channel.fromList(params.alphabet_size))
                               .combine(channel.fromList(params.seq_seed))
                               .combine(channel.fromList(params.prior_seed))

    simulations = parameter_channel | map { ncells, nchars, alphabet, seq_seed, prior_seed ->
        id               = "k${nchars}M${alphabet}p${prior_seed}_medium_sub${ncells}_r${seq_seed}"
        prefix           = "${params.simulation_dir}/${id}"
        character_matrix = "${prefix}/character_matrix.csv"
        tree             = "${prefix}/tree.nwk"
        [id, character_matrix, tree]
    }

    simulations | fast_laml
    simulations | laml
}
