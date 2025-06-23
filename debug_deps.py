# debug_deps.py
import os
import pkg_resources
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- 核心调试代码 ---
print("--- Analyzing Package Sizes ---")
pkgs = sorted([(p.project_name, p.location) for p in pkg_resources.working_set], key=lambda x: x[0])
total = 0
for name, loc in pkgs:
    try:
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(loc) for f in fs)
        print(f'{name:<30} {size / 1024 / 1024:>10.2f} MB')
        total += size
    except Exception as e:
        print(f"Could not calculate size for {name}: {e}")
print(f'--- Total Size: {total / 1024 / 1024:.2f} MB ---')
print("--- Debug script finished. Deployment will now fail as expected. Check logs for sizes. ---")
# --- 核心调试代码结束 ---


# --- 创建一个假的 HTTP 服务器以满足 Vercel 构建要求 ---
# Vercel 的 Python 构建器需要一个符合 WSGI 规范的 "app" 对象。
# 我们提供一个最简单的对象，它什么都不做，但能让构建过程通过。
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/plain')
        self.end_headers()
        self.wfile.write(b'Debug script ran successfully. Check build logs.')

# Vercel 会寻找这个 'app' 变量
app = handler 

# 这部分代码实际上不会在构建时运行，但对于满足构建器规范是必要的。
if __name__ == '__main__':
    # 这只是为了本地测试，Vercel 不会执行这里
    with HTTPServer(('', 8000), app) as server:
        server.serve_forever()
