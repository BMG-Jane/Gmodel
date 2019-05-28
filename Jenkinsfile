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
	sh 'docker build https://github.com/BostonMeditechGroup/calcification_detection.git#dev:scripts -t egret:gm1.0 --build-arg username=gautham1994 --build-arg password=Nandanam94'

      }
    }
    stage('Run the Docker') {
      steps {
	sh 'nvidia-docker run -d -it --rm --name gmodel egret:gm1.0'
	sh 'docker cp ./gm_temp/. gmodel:/home/scripts'
	sh 'docker commit calc egret:gm1.0'
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