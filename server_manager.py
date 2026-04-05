import os
import signal
import socket
import subprocess
import time
from pathlib import Path


class ServerManager:
    def __init__(self, project_root: str | Path | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parent)
        self.processes: list[subprocess.Popen] = []

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            return sock.connect_ex((host, port)) == 0

    def _wait_for_port(
        self,
        proc: subprocess.Popen,
        host: str,
        port: int,
        timeout: int,
        script_name: str,
    ) -> None:
        deadline = time.time() + timeout

        while time.time() < deadline:
            # если процесс уже умер — значит старт не удался
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Скрипт {script_name} завершился раньше времени "
                    f"с кодом {proc.returncode}"
                )

            if self._is_port_open(host, port):
                print(f"[scripts] {script_name} успешно запущен на {host}:{port}")
                return

            time.sleep(1)

        raise TimeoutError(
            f"Таймаут ожидания запуска {script_name} "
            f"(порт {host}:{port} не открылся за {timeout} сек.)"
        )

    def start_script(
        self,
        script_rel_path: str,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout: int = 120,
    ) -> subprocess.Popen | None:
        script_path = (self.project_root / script_rel_path).resolve()

        if not script_path.exists():
            raise FileNotFoundError(f"Не найден скрипт: {script_path}")

        # если нужный порт уже занят, считаем что сервер уже поднят извне
        if port is not None and self._is_port_open(host, port):
            print(
                f"[scripts] Порт {port} уже занят, "
                f"пропускаю запуск {script_rel_path}"
            )
            return None

        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        log_file = logs_dir / f"{script_path.stem}.log"
        log_handle = open(log_file, "a", encoding="utf-8")

        proc = subprocess.Popen(
            ["bash", str(script_path)],
            cwd=str(self.project_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # отдельная process group для Linux
        )

        # сохраним handle, чтобы лог не закрылся раньше времени
        proc._graf_log_handle = log_handle  # type: ignore[attr-defined]
        self.processes.append(proc)

        print(f"[scripts] Запускаю {script_rel_path}...")

        if port is not None:
            self._wait_for_port(proc, host, port, startup_timeout, script_rel_path)

        return proc

    def start_default_servers(self) -> None:
        self.start_script("scripts/start_server.sh", port=8000, startup_timeout=180)
        self.start_script("scripts/start_server_gemma.sh", port=8001, startup_timeout=180)

    def stop_all(self) -> None:
        # сначала мягкая остановка
        for proc in reversed(self.processes):
            if proc.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.time() + 10
        while time.time() < deadline:
            alive = [p for p in self.processes if p.poll() is None]
            if not alive:
                break
            time.sleep(0.2)

        # если кто-то не умер — добиваем
        for proc in reversed(self.processes):
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        # закрываем логи
        for proc in self.processes:
            handle = getattr(proc, "_graf_log_handle", None)
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass

        self.processes.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_all()