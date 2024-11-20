server {
    listen 80;
    server_name api.example.com;  # Replace 'api.example.com' dynamically

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_buffering off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}