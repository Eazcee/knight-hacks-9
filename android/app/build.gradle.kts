import java.util.Properties

plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.compose)
}

val localProperties: Properties =
    Properties().apply {
      rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
    }

// "0" is the documented Developer Mode placeholder; attestation is skipped in Dev Mode.
val mwdatApplicationId: String = localProperties.getProperty("mwdat_application_id", "0")
val mwdatClientToken: String = localProperties.getProperty("mwdat_client_token", "0")

android {
  namespace = "com.beacon.app"
  compileSdk {
    version =
        release(36) {
          minorApiLevel = 1
        }
  }

  defaultConfig {
    applicationId = "com.beacon.app"
    minSdk = 31
    targetSdk = 36
    versionCode = 1
    versionName = "1.0"

    manifestPlaceholders["mwdat_application_id"] = mwdatApplicationId
    manifestPlaceholders["mwdat_client_token"] = mwdatClientToken
  }

  buildTypes {
    release {
      optimization {
        enable = false
      }
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }
  buildFeatures {
    compose = true
  }
}

dependencies {
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.activity.compose)
  implementation(libs.androidx.compose.material3)
  implementation(libs.androidx.compose.ui)
  implementation(libs.androidx.compose.ui.tooling.preview)
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.lifecycle.runtime.compose)
  implementation(libs.androidx.lifecycle.viewmodel.compose)
  implementation(libs.okhttp)
  implementation(libs.play.services.location)
  implementation(libs.mwdat.core)
  implementation(libs.mwdat.camera)
  implementation(libs.mwdat.display)
  debugImplementation(libs.androidx.compose.ui.tooling)
}
