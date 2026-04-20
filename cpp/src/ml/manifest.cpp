#include "ml/manifest.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

std::vector<std::string> FeatureManifest::feature_names() const {
    std::vector<std::string> out;
    out.reserve(features.size());
    for (const auto& f : features) out.push_back(f.name);
    return out;
}

FeatureManifest load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Cannot open manifest: " + path);
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("Invalid JSON in manifest " + path + ": " + e.what());
    }

    auto require = [&](const char* key) {
        if (!j.contains(key)) {
            throw std::runtime_error(
                "Manifest " + path + " missing required key: " + key);
        }
    };
    require("n_features");
    require("model_version");
    require("destroy_ops");
    require("repair_ops");

    FeatureManifest m;
    m.n_features = j["n_features"].get<int>();
    m.model_version = j["model_version"].get<std::string>();
    m.destroy_ops = j["destroy_ops"].get<std::vector<std::string>>();
    m.repair_ops = j["repair_ops"].get<std::vector<std::string>>();

    // Detect format: NEW manifests carry `features` (objects) along with `mode`
    // and `batches`. LEGACY manifests have only `feature_order` (bare strings).
    const bool has_new = j.contains("features") && j.contains("mode");

    if (has_new) {
        m.is_legacy = false;
        m.mode = parse_feature_mode(j["mode"].get<std::string>());
        if (j.contains("batches")) {
            m.batches = j["batches"].get<std::vector<std::string>>();
        }

        for (const auto& fj : j["features"]) {
            if (!fj.contains("name")) {
                throw std::runtime_error("Manifest " + path
                    + ": feature entry missing 'name'");
            }
            ManifestFeature mf;
            mf.name = fj["name"].get<std::string>();
            mf.batch = fj.value("batch", std::string{});
            std::string mode_str = fj.value("mode", std::string{"both"});
            mf.mode = parse_feature_mode(mode_str);
            mf.tolerance = fj.value("tolerance", 0.0);
            m.features.push_back(std::move(mf));
        }
    } else {
        // Legacy path: features are bare strings in `feature_order`. Tags unknown.
        require("feature_order");
        m.is_legacy = true;
        auto names = j["feature_order"].get<std::vector<std::string>>();
        for (const auto& n : names) {
            ManifestFeature mf;
            mf.name = n;
            mf.batch = "";              // unknown
            mf.mode = FeatureMode::Both;// permissive — caller skips tag check
            mf.tolerance = 0.0;
            m.features.push_back(std::move(mf));
        }
    }

    if (static_cast<int>(m.features.size()) != m.n_features) {
        std::ostringstream oss;
        oss << "Manifest " << path << ": features list has "
            << m.features.size() << " entries but n_features="
            << m.n_features;
        throw std::runtime_error(oss.str());
    }

    return m;
}
