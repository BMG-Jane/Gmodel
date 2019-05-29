pipeline {
  agent any
  stages {
    stage('Start'){
      steps {
        sh 'ls -la'
        sh 'pwd'
        slackSend (color: '#FFFF00', message: "STARTED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
      }
    }

    stage('Build the Docker') {
      steps {
        sh 'ls'
	sh 'docker build https://github.com/BMG-Jane/Gmodel.git#master:scripts -t egret:jane1.0 --build-arg username=gautham1994 --build-arg password=Nandanam94'

      }
    }
    stage('Run the Docker') {
      steps {
	sh 'nvidia-docker run -d -it --rm --name gmodel egret:jane1.0'
	sh 'docker cp ./gmodel_temp/. gmodel:/home/scripts/gmodel_tmp/'
	sh 'docker commit gmodel egret:jane1.0'
      }
    }
    stage('Testing') {
      steps {
        sh 'nvidia-docker exec -i gmodel python3 /home/scripts/GMcall.py'
      }
    }
  }
    post {
    success {
      slackSend (color: '#00FF00', message: "SUCCESSFUL: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
    }

    failure {
      slackSend (color: '#FF0000', message: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
    }
  }

}
