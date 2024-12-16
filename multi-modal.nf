if (nextflow.version.matches(">=20.07.1")){
    nextflow.enable.dsl=2
}else{
    // Support lower version of nextflow
    nextflow.preview.dsl=2
}


projectDir = workflow.projectDir

params.path_test_fMRI = "/home/lining/xmn/model/SCZ/GAD/56_new/fMRI.csv"
params.path_test_sMRI = "/home/lining/xmn/model/SCZ/GAD/56_new/sMRI.csv"
params.judge = "y"


workflow {

    depression()
}


process exitCheck {
    output:
    val ' '
    if (file("${params.path_test_fMRI}").exist() & file("${params.path_test_sMRI}").exist())) {

        exit 1

    }

    """
    echo error
    """
}
process depression {

    script:
    if (params.judge == "Main")
    {
    println "Predict with Graph network"
    println "${params.path_test_fMRI}"
    println "${params.path_test_sMRI}"
    println params.judge

    }
    else {
    println "Predict with RF"
    }

    """
    python ${projectDir}/main_SCZ_dynamic_edges_dense_find_seed_COBRE.py \
            --path_test_fMRI ${params.path_test_fMRI} \
            --path_test_sMRI ${params.path_test_sMRI} \
            --judge ${params.judge}
    """
}