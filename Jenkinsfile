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
                sh 'docker run air-quality-pipeline'
            }
        }
    }
}

