# Deploy demo

## Chạy bằng Docker (khuyên dùng)

Image chứa CODE (~9.4 GB gồm torch+CUDA); weights 19 GB, `.env`, certs, logs đều
mount từ host — không bake vào image. Yêu cầu host có NVIDIA driver +
nvidia-container-toolkit (máy này đã có runtime `nvidia` + CDI).

```bash
cd /home/ubuntu/ai-assistant
docker compose up -d --build      # build lần đầu ~5 phút
docker compose logs -f            # đợi /health trả "status":"ok" (~2 phút load model)
```

Mở trên điện thoại: `https://<IP-may-chu>:8443/` (cert tự ký → Advanced → Proceed).
Dừng: `docker compose down`. Xem GPU: `nvidia-smi` sẽ thấy process python của
container dùng ~11.3 GiB.

Đã kiểm chứng trong container: `/health` ok, WSS `/v1/vision` inference thật
(2 frame pole_00 liên tiếp → frame 2 phát OBSTACLE_FRONT đúng luật confirm=2),
latency warm ~1.2 s/frame.

Lưu ý build: image cần `gcc` + `libc6-dev` cho triton JIT compile lúc runtime —
thiếu là mọi request vision trả lỗi "Failed to find C compiler"/"stdlib.h".

## Chạy nhanh không Docker (không cần cài gì)

```bash
/home/ubuntu/ai-assistant/scripts/run_demo_server.sh     # HTTPS 0.0.0.0:8443
```

Mở trên điện thoại: `https://<IP-may-chu>:8443/` — trình duyệt sẽ cảnh báo cert
tự ký, chọn "Advanced → Proceed" một lần. Camera/mic chỉ hoạt động trên HTTPS.

Yêu cầu mạng: điện thoại phải tới được port 8443 của server (cùng VPN/tunnel,
security group mở TCP 8443). Instance hiện không có public IPv4 — truy cập qua
VPN (Tailscale/WireGuard) hoặc SSH tunnel:

```bash
# trên máy trung gian có IP public:
ssh -L 8443:127.0.0.1:8443 ubuntu@<server>
# điện thoại truy cập https://<IP-may-trung-gian>:8443  (chạy lại script với HOST=0.0.0.0)
```

## Chạy như service (systemd)

```bash
sudo cp /home/ubuntu/ai-assistant/deploy/ai-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-assistant
journalctl -u ai-assistant -f          # xem log
```

## Cert tự ký

Script tự tạo nếu chưa có (`certs/cert.pem`, `certs/key.pem`). Muốn thêm SAN
(IP mới/domain) thì xoá 2 file đó và tạo lại bằng openssl với
`-addext "subjectAltName=..."`.

## Lưu ý proxy

`run_demo_server.sh` bỏ mọi biến proxy trước khi khởi động: proxy của máy này
làm hỏng hub-check của HuggingFace kể cả khi model đã cache đầy đủ (README,
mục tải model). Kết nối trực tiếp đã xác nhận OK cho cả HF lẫn Google AI
(Gemini describe chạy qua đường trực tiếp này).
