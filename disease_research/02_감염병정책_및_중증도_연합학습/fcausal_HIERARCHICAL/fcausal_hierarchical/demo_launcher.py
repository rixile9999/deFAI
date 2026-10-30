"""
CES 2026 데모 녹화용 통합 런처
- 4K 주모니터 (3840x2160) 최적화
- 7개 터미널 창을 격자 배치
- 연합학습 자동 실행
- 시각화 영상 자동 재생
"""

import subprocess
import time
import os
import sys
from pathlib import Path

# Windows API for window positioning
try:
    import win32gui
    import win32con
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
    print("Warning: pywin32 not installed. Window positioning will be limited.")
    print("Install with: pip install pywin32")

class DemoLauncher:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        local_python = self.project_dir / ".venv-policy-research" / "Scripts" / "python.exe"
        self.python_exe = str(local_python) if local_python.exists() else sys.executable

        # 4K 주모니터 해상도
        self.screen_width = 3840
        self.screen_height = 2160

        # 터미널 창 크기 (7개를 격자로 배치)
        # 3x3 격자 (7개 사용, 2개 빈칸)
        self.terminal_width = self.screen_width // 3
        self.terminal_height = self.screen_height // 3

        # 프로세스 저장
        self.processes = {}
        self.window_handles = {}

    def print_banner(self, text):
        """예쁜 배너 출력"""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80 + "\n")

    def start_terminal(self, name, command, position):
        """새 터미널 창 시작 및 위치 지정"""
        x, y = position

        # PowerShell 또는 CMD로 새 창 실행
        # -NoExit: 명령 실행 후에도 창 유지
        ps_command = f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd \'{self.project_dir}\'; $env:PYTHONIOENCODING=\'utf-8\'; {command}" -WindowStyle Normal'

        proc = subprocess.Popen(
            ["powershell", "-Command", ps_command],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        self.processes[name] = proc

        # 창 위치 조정 (약간의 지연 필요)
        if HAS_WIN32:
            time.sleep(2.5)  # 창이 생성될 때까지 대기 (증가)
            try:
                self._move_window(name, x, y, self.terminal_width, self.terminal_height)
            except Exception as e:
                print(f"⚠️  {name} 창 배치 실패 (수동으로 배치하세요): {e}")

        return proc

    def _move_window(self, name, x, y, width, height):
        """Windows API로 창 위치/크기 조정"""
        if not HAS_WIN32:
            return

        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                # PowerShell 창 찾기
                if "powershell" in title.lower() or "Windows PowerShell" in title:
                    windows.append(hwnd)
            return True

        windows = []
        win32gui.EnumWindows(callback, windows)

        # 가장 최근에 생성된 창 (마지막 창)
        if windows:
            hwnd = windows[-1]

            # 창 핸들 유효성 검증
            if not win32gui.IsWindow(hwnd):
                print(f"⚠️  {name}: 유효하지 않은 창 핸들")
                return

            self.window_handles[name] = hwnd

            # 창 스타일 변경 (최대화 해제)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.3)

            # 위치 및 크기 설정
            win32gui.MoveWindow(hwnd, x, y, width, height, True)
            print(f"✅ {name} 창 배치: ({x}, {y}) {width}x{height}")
        else:
            print(f"⚠️  {name}: PowerShell 창을 찾을 수 없음 (수동 배치 필요)")

    def launch_tier0(self):
        """Tier 0 (Central) 실행"""
        self.print_banner("🚀 Tier 0: KDCA Central Government")

        # 위치: 상단 중앙
        pos = (self.terminal_width, 0)

        cmd = f'"{self.python_exe}" run_tier0.py'
        return self.start_terminal("Tier 0 (KDCA)", cmd, pos)

    def launch_tier1_seoul(self):
        """Tier 1: Seoul Province"""
        self.print_banner("🏙️ Tier 1: Seoul Province (Treatment)")

        # 위치: 중단 좌측
        pos = (0, self.terminal_height)

        cmd = f'"{self.python_exe}" run_tier1.py --port 9001 --region Seoul --treated --min-districts 2'
        return self.start_terminal("Tier 1 (Seoul)", cmd, pos)

    def launch_tier1_gyeonggi(self):
        """Tier 1: Gyeonggi Province"""
        self.print_banner("🌆 Tier 1: Gyeonggi Province (Control)")

        # 위치: 중단 우측
        pos = (self.terminal_width * 2, self.terminal_height)

        cmd = f'"{self.python_exe}" run_tier1.py --port 9002 --region Gyeonggi --min-districts 2'
        return self.start_terminal("Tier 1 (Gyeonggi)", cmd, pos)

    def launch_tier2_gangnam(self):
        """Tier 2: Gangnam District"""
        self.print_banner("📍 Tier 2: Gangnam District")

        # 위치: 하단 좌측 상단
        pos = (0, self.terminal_height * 2)

        cmd = f'"{self.python_exe}" run_tier2.py --data data/districts/seoul_gangnam.xlsx --target 127.0.0.1:9001'
        return self.start_terminal("Tier 2 (Gangnam)", cmd, pos)

    def launch_tier2_seocho(self):
        """Tier 2: Seocho District"""
        self.print_banner("📍 Tier 2: Seocho District")

        # 위치: 하단 좌측 하단
        pos = (self.terminal_width, self.terminal_height * 2)

        cmd = f'"{self.python_exe}" run_tier2.py --data data/districts/seoul_seocho.xlsx --target 127.0.0.1:9001'
        return self.start_terminal("Tier 2 (Seocho)", cmd, pos)

    def launch_tier2_suwon(self):
        """Tier 2: Suwon City"""
        self.print_banner("📍 Tier 2: Suwon City")

        # 위치: 하단 우측 상단
        pos = (self.terminal_width * 2, self.terminal_height * 2)

        cmd = f'"{self.python_exe}" run_tier2.py --data data/districts/gyeonggi_suwon.xlsx --target 127.0.0.1:9002'
        return self.start_terminal("Tier 2 (Suwon)", cmd, pos)

    def launch_tier2_seongnam(self):
        """Tier 2: Seongnam City"""
        self.print_banner("📍 Tier 2: Seongnam City")

        # 위치: 상단 우측
        pos = (self.terminal_width * 2, 0)

        cmd = f'"{self.python_exe}" run_tier2.py --data data/districts/gyeonggi_seongnam.xlsx --target 127.0.0.1:9002'
        return self.start_terminal("Tier 2 (Seongnam)", cmd, pos)

    def play_visualization(self):
        """시각화 영상 재생"""
        self.print_banner("🎬 Playing Visualization Video")

        video_path = self.project_dir / "results" / "policy_analysis_current.mp4"

        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            print("Generating visualization first...")
            subprocess.run([self.python_exe, "demo1_hierarchical_visualization.py"])

        # Windows Media Player 또는 기본 비디오 플레이어로 재생
        os.startfile(str(video_path))
        print(f"✅ Opened: {video_path}")

    def show_layout_guide(self):
        """화면 배치 가이드 표시"""
        print("\n" + "="*80)
        print("  📺 SCREEN LAYOUT (3840x2160)")
        print("="*80)
        print("""
        ┌─────────────────┬─────────────────┬─────────────────┐
        │                 │   Tier 0 KDCA   │ Tier 2 Seongnam │
        │     (Empty)     │    (Central)    │  (Gyeonggi)     │
        │                 │                 │                 │
        ├─────────────────┼─────────────────┼─────────────────┤
        │  Tier 1 Seoul   │                 │ Tier 1 Gyeonggi │
        │   (Treatment)   │    (Empty)      │    (Control)    │
        │                 │                 │                 │
        ├─────────────────┼─────────────────┼─────────────────┤
        │ Tier 2 Gangnam  │ Tier 2 Seocho   │  Tier 2 Suwon   │
        │    (Seoul)      │    (Seoul)      │   (Gyeonggi)    │
        │                 │                 │                 │
        └─────────────────┴─────────────────┴─────────────────┘

        Each cell: 1280x720 pixels
        """)
        print("="*80 + "\n")

    def run_sequential_demo(self):
        """순차적 데모 실행 (단계별 설명)"""
        self.print_banner("🎯 CES 2026 DEMO - Sequential Launch")
        self.show_layout_guide()

        print("\n🔄 Starting in 3 seconds...")
        time.sleep(3)

        # Step 1: Tier 0 시작
        input("\n📍 Step 1: Press ENTER to launch Tier 0 (KDCA Central)...")
        self.launch_tier0()
        time.sleep(2)

        # Step 2: Tier 1 시작
        input("\n📍 Step 2: Press ENTER to launch Tier 1 (Provinces)...")
        self.launch_tier1_seoul()
        time.sleep(1)
        self.launch_tier1_gyeonggi()
        time.sleep(2)

        # Step 3: Tier 2 시작
        input("\n📍 Step 3: Press ENTER to launch Tier 2 (Districts)...")
        self.launch_tier2_gangnam()
        time.sleep(0.5)
        self.launch_tier2_seocho()
        time.sleep(0.5)
        self.launch_tier2_suwon()
        time.sleep(0.5)
        self.launch_tier2_seongnam()
        time.sleep(2)

        # Step 4: 시각화 재생
        input("\n📍 Step 4: Press ENTER to play visualization video...")
        self.play_visualization()

        self.print_banner("✅ Demo Launched Successfully!")
        print("💡 Tip: Keep all windows visible during screen recording")
        print("📹 Start your screen recording software now!")
        print("\n⚠️  Press Ctrl+C to stop all processes when done\n")

        # 프로세스 모니터링
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping all processes...")
            self.cleanup()

    def run_auto_demo(self):
        """자동 데모 실행 (녹화 즉시 시작)"""
        self.print_banner("🎯 CES 2026 DEMO - Auto Launch")
        self.show_layout_guide()

        print("\n⏰ Auto-launching in 5 seconds...")
        print("📹 Start your screen recording NOW!")
        time.sleep(5)

        # Step 1: Tier 0 시작
        self.print_banner("STEP 1: Launching Tier 0 (Central Server)")
        self.launch_tier0()
        print("⏳ Waiting 3 seconds for server to be ready...")
        time.sleep(3)

        # Step 2: Tier 1 시작
        self.print_banner("STEP 2: Launching Tier 1 (Provincial Servers)")
        self.launch_tier1_seoul()
        print("⏳ Waiting 2 seconds...")
        time.sleep(2)

        self.launch_tier1_gyeonggi()
        print("⏳ Waiting 5 seconds for servers to open ports...")
        time.sleep(5)

        # Step 3: Tier 2 시작
        self.print_banner("STEP 3: Launching Tier 2 (District Clients)")
        print("📍 Connecting districts to provincial servers...")

        self.launch_tier2_gangnam()
        time.sleep(1)
        self.launch_tier2_seocho()
        time.sleep(1)
        self.launch_tier2_suwon()
        time.sleep(1)
        self.launch_tier2_seongnam()

        print("\n✅ All 7 terminals launched!")
        print("⏳ Waiting 20 seconds for federated learning to complete...")
        time.sleep(20)

        # 시각화 자동 재생
        self.print_banner("Playing Visualization Video")
        self.play_visualization()

        self.print_banner("✅ Demo Running!")
        print("📊 All windows are now running")
        print("⚠️  Press Ctrl+C to stop when recording is done\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping all processes...")
            self.cleanup()

    def cleanup(self):
        """모든 프로세스 종료"""
        for name, proc in self.processes.items():
            try:
                proc.terminate()
                print(f"✅ Stopped: {name}")
            except:
                pass

        print("\n✅ All processes stopped.")
        print("📁 Output files:")
        print(f"   - demo1_hierarchical.mp4 (시각화 영상)")
        print(f"   - logs/ (연합학습 로그)")

def main():
    """메인 실행 함수"""
    launcher = DemoLauncher()

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🎬 CES 2026 DEMO LAUNCHER                                  ║
║   Hierarchical Federated Learning with DiD Analysis          ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    print("\nSelect demo mode:")
    print("  1. Sequential Launch (Step-by-step with prompts)")
    print("  2. Auto Launch (Immediate start for recording)")
    print("  3. Quick Test (Just play visualization)")
    print("  4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    if choice == "1":
        launcher.run_sequential_demo()
    elif choice == "2":
        launcher.run_auto_demo()
    elif choice == "3":
        launcher.play_visualization()
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice!")
        sys.exit(1)

if __name__ == "__main__":
    main()
