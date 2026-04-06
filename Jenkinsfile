pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building project...'
            }
        }

        stage('Run Docker') {
            steps {
                sh 'docker build -t air-quality-pipeline .'

                withCredentials([string(credentialsId: 'azure-conn', variable: 'AZURE_CONN_STR')]) {
                    sh 'docker run -e AZURE_CONN_STR=$AZURE_CONN_STR air-quality-pipeline'
                }
            }
        }
    }
}
