terraform {
  required_providers {
    google = {}
  }
}

resource "google_secret_manager_secret" "github_token_secret" {
  project   = PROJECT_ID
  secret_id = SECRET_ID

  replication {
    # automatic = true
  }
}

resource "google_secret_manager_secret_version" "github_token_secret_version" {
  secret      = google_secret_manager_secret.github_token_secret.id
  secret_data = GITHUB_PAT
}

data "google_iam_policy" "serviceagent_secretAccessor" {
  binding {
    role    = "roles/secretmanager.secretAccessor"
    members = ["serviceAccount:service-PROJECT_NUMBER@gcp-sa-cloudbuild.iam.gserviceaccount.com"]
  }
}

resource "google_secret_manager_secret_iam_policy" "policy" {
  project     = google_secret_manager_secret.github_token_secret.project
  secret_id   = google_secret_manager_secret.github_token_secret.secret_id
  policy_data = data.google_iam_policy.serviceagent_secretAccessor.policy_data
}

resource "google_cloudbuildv2_connection" "my_connection" {
  project  = PROJECT_ID
  location = REGION
  name     = CONNECTION_NAME

  github_config {
    app_installation_id = INSTALLATION_ID
    authorizer_credential {
      oauth_token_secret_version = google_secret_manager_secret_version.github_token_secret_version.id
    }
  }
  depends_on = [google_secret_manager_secret_iam_policy.policy]
}

resource "google_cloudbuildv2_repository" "my_repository" {
  project           = "PROJECT_ID"
  location          = "REGION"
  name              = "REPO_NAME"
  parent_connection = google_cloudbuildv2_connection.my_connection.name
  remote_uri        = "URI"
}
