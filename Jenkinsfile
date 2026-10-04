pipeline {
    agent { label 'production' }

    environment {
        PATH = "/home/lcastaa/.local/bin:${env.PATH}"
        NEUROMEM_HOST_IP = '192.168.1.208'
    }

    options {
        disableConcurrentBuilds()
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {
        stage('Validate') {
            steps {
                sh 'uv sync --frozen --all-groups'
                sh 'uv run ruff check .'
                sh 'uv run ruff format --check .'
            }
        }

        stage('Unit tests') {
            steps {
                sh 'uv run pytest tests/unit'
            }
        }

        stage('Deploy') {
            steps {
                withCredentials([
                    string(credentialsId: 'neuromem-openai-api-key', variable: 'OPENAI_API_KEY'),
                    string(credentialsId: 'neuromem-rabbitmq-password', variable: 'RABBITMQ_PASSWORD')
                ]) {
                    sh '''
                        set +x
                        umask 077
                        trap 'rm -f .env' EXIT
                        printf 'OPENAI_API_KEY=%s\\nRABBITMQ_USER=neuromem\\nRABBITMQ_PASSWORD=%s\\n' "$OPENAI_API_KEY" "$RABBITMQ_PASSWORD" > .env
                        docker compose -p neuromem config --quiet
                        docker compose -p neuromem up --build --detach --remove-orphans

                        for attempt in $(seq 1 60); do
                            if curl -fsS "http://${NEUROMEM_HOST_IP}:8100/health" >/dev/null; then
                                docker compose -p neuromem ps
                                exit 0
                            fi
                            sleep 5
                        done

                        docker compose -p neuromem logs --tail=100
                        exit 1
                    '''
                }
            }
        }
    }

    post {
        always {
            deleteDir()
        }
    }
}
