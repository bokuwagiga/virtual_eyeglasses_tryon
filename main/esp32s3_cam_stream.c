#include "esp_camera.h"
#include "esp_log.h"
#include "esp_http_server.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "freertos/event_groups.h"
#include "esp_netif.h"
#include "esp_mac.h"

#define WIFI_SSID "your_wifi_ssid"
#define WIFI_PASS "your_wifi_password"

static const char *TAG = "cam_stream";

// MJPEG stream handler
esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;

    res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
    if (res != ESP_OK) return res;

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            continue;
        }

        char part_buf[64];
        snprintf(part_buf, sizeof(part_buf),
                 "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
                 fb->len);

        res = httpd_resp_send_chunk(req, part_buf, strlen(part_buf));
        res |= httpd_resp_send_chunk(req, (char *)fb->buf, fb->len);
        res |= httpd_resp_send_chunk(req, "\r\n", 2);

        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;

        vTaskDelay(30 / portTICK_PERIOD_MS); // ~30 fps
    }

    return res;
}

// Root page handler
esp_err_t root_handler(httpd_req_t *req) {
    const char* resp_str = "<html><body><h1>ESP32-CAM</h1><img src=\"/mjpeg\"></body></html>";
    httpd_resp_send(req, resp_str, strlen(resp_str));
    return ESP_OK;
}

// Start HTTP server
httpd_handle_t start_webserver() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t stream_uri = {
            .uri = "/mjpeg",
            .method = HTTP_GET,
            .handler = stream_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &stream_uri);

        // Root page handler
        httpd_uri_t root_uri = {
            .uri = "/",
            .method = HTTP_GET,
            .handler = root_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &root_uri);
    }

    return server;
}

// Initialize and start the camera
void start_camera() {
    camera_config_t config = {
        .pin_pwdn = -1,
        .pin_reset = -1,
        .pin_xclk = 10,
        .pin_sccb_sda = 40,
        .pin_sccb_scl = 39,

        .pin_d7 = 48,
        .pin_d6 = 11,
        .pin_d5 = 12,
        .pin_d4 = 14,
        .pin_d3 = 16,
        .pin_d2 = 18,
        .pin_d1 = 17,
        .pin_d0 = 15,

        .pin_vsync = 38,
        .pin_href = 47,
        .pin_pclk = 13,

        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,

        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QVGA,
        .jpeg_quality = 12,
        .fb_count = 3,
        .grab_mode = CAMERA_GRAB_LATEST,
        .fb_location = CAMERA_FB_IN_PSRAM
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s->id.PID == OV3660_PID) {
        s->set_vflip(s, 1);
        s->set_brightness(s, 1);
        s->set_saturation(s, -2);
    }
    s->set_framesize(s, FRAMESIZE_QVGA); // Smoother stream
}

// Connect to Wi-Fi
void wifi_init_sta() {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Connecting to WiFi SSID:%s", WIFI_SSID);

    int retry = 0;
    while (retry < 20 && esp_wifi_connect() != ESP_OK) {
        ESP_LOGW(TAG, "Retrying WiFi...");
        vTaskDelay(500 / portTICK_PERIOD_MS);
        retry++;
    }

    while (true) {
        wifi_ap_record_t ap_info;
        if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
            ESP_LOGI(TAG, "Connected to SSID:%s", ap_info.ssid);
            break;
        }
        ESP_LOGW(TAG, "Waiting for WiFi connection...");
        vTaskDelay(500 / portTICK_PERIOD_MS);
    }

    // Print IP
    esp_netif_ip_info_t ip_info;
    esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    if (netif == NULL) {
        ESP_LOGE(TAG, "Failed to get netif handle");
        return;
    }
    esp_netif_get_ip_info(netif, &ip_info);
    ESP_LOGI(TAG, "Camera Ready! Stream: http://" IPSTR "/mjpeg", IP2STR(&ip_info.ip));
}

// Main function
void app_main() {
    ESP_ERROR_CHECK(nvs_flash_init());
    wifi_init_sta();
    start_camera();
    start_webserver();
}
