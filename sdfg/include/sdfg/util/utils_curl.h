
#pragma once
#include <curl/curl.h>
#include <string>

struct HttpResult {
    CURLcode curl_code;
    long http_status;
    std::string body;
    std::string error_message;
};

HttpResult
post_json(CURL* curl, const std::string& url, const std::string& json_payload, struct curl_slist* headers = nullptr);
