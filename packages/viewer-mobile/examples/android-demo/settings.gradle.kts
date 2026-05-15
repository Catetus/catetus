pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "splatforge-android-demo"
include(":app")
include(":splatforge")
project(":splatforge").projectDir = file("../../android/splatforge")
