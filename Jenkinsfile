pipeline {
    agent { label 'development' }

    environment {
        PATH = "/home/lcastaa/.local/bin:${env.PATH}"
    }

    options {
        disableConcurrentBuilds()
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
    }

    parameters {
        booleanParam(
            name: 'DEPLOY',
            defaultValue: false,
            description: 'Deploy the tested revision to the Docker host.'
        )
    }

    stages {
        stage('Validate') {
            steps {
                sh 'uv sync --frozen --all-groups'
                sh 'uv run ruff check .'
                sh 'uv run ruff format --check .'
                sh 'uv run mypy memory models apps'
            }
        }

        stage('Unit tests') {
            steps {
                sh 'uv run pytest tests/unit'
            }
        }

        stage('Deploy') {
            when {
                expression { params.DEPLOY }
            }
            steps {
                withCredentials([string(credentialsId: 'neuromem-openai-api-key', variable: 'OPENAI_API_KEY')]) {
                    sh '''
                        set +x
                        umask 077
                        trap 'rm -f .env' EXIT
                        printf 'OPENAI_API_KEY=%s\\n' "$OPENAI_API_KEY" > .env
                        docker compose -p neuromem config --quiet
                        docker compose -p neuromem up --build --detach --remove-orphans

                        for attempt in $(seq 1 60); do
                            if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
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
