#include "ml/manifest.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

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
    require("feature_order");
    require("n_features");
    require("model_version");
    require("destroy_ops");
    require("repair_ops");

    FeatureManifest m;
    m.feature_order = j["feature_order"].get<std::vector<std::string>>();
    m.n_features = j["n_features"].get<int>();
    m.model_version = j["model_version"].get<std::string>();
    m.destroy_ops = j["destroy_ops"].get<std::vector<std::string>>();
    m.repair_ops = j["repair_ops"].get<std::vector<std::string>>();

    if (static_cast<int>(m.feature_order.size()) != m.n_features) {
        std::ostringstream oss;
        oss << "Manifest " << path << ": feature_order has "
            << m.feature_order.size() << " entries but n_features="
            << m.n_features;
        throw std::runtime_error(oss.str());
    }

    return m;
}
