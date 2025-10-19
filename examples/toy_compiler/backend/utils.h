#pragma once

#include <string>
#include <vector>

#include "common/status.h"

namespace toy::utils {

bool getBoolEnv(const std::string &env);

std::string getStrEnv(const std::string &env);

std::vector<std::string> splitStringBySpace(const std::string& str);

std::string genTempFile();
std::string readFile(const std::string& filePath);
bool writeFile(const std::string& data, const std::string& filePath);

status::Result<bool> runCommand(const std::string& cmd, std::string& stdoutOutput, std::string& stderrOutput);

} // namespace toy::utils