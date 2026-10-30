"""
계층형 F-Causal 시각화 (3-Tier Hierarchical Structure)

결과는 results/latest_policy_analysis.json에서 읽는다.
그래프는 집단별 전후 평균을 잇는 도식이며 실제 일별 추세가 아니다.
과거 영상·시연 문서의 고정 효과·유의성 수치는 검증 결과가 아니다.

실제 지역 구조:
- Tier 2: 강남구, 서초구 (서울) / 수원시, 성남시 (경기)
- Tier 1: 서울시 (Treatment), 경기도 (Control)
- Tier 0: KDCA (중앙정부)
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# 영문 폰트 설정 (CES 국제 전시용)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

def create_demo1_hierarchical():
    """3-Tier 계층형 구조 및 DiD 분석 시각화 (CES 국제 전시용)"""

    root = Path(__file__).resolve().parent
    result = json.loads((root / 'results/latest_policy_analysis.json').read_text(encoding='utf-8'))
    def group_mean(arm, period):
        provinces = [p for p in result['provinces'] if p['treated'] == arm]
        return sum(p[period]['mean'] * p['n_districts'] for p in provinces) / sum(p['n_districts'] for p in provinces)
    control_pre, control_post = group_mean(0, 'pre'), group_mean(0, 'post')
    treated_pre, treated_post = group_mean(1, 'pre'), group_mean(1, 'post')

    # 1. Canvas Setup
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1, 5.5)
    ax.axis('off')

    # 배경색 (학술적 화이트)
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    # 자물쇠 이미지 로드
    try:
        lock_img = Image.open('lock.png')
        lock_img = lock_img.resize((40, 40), Image.Resampling.LANCZOS)
    except:
        lock_img = None
        print("Warning: lock.png not found. Using text fallback.")

    # 2. Node Positions (실제 지역 반영)
    # Tier 0: Central Government (KDCA)
    pos_t0 = (0, 4.5)

    # Tier 1: Provinces
    pos_seoul = (-2, 2.5)      # 서울시 (Treatment)
    pos_gyeonggi = (2, 2.5)    # 경기도 (Control)
    pos_t1 = [pos_seoul, pos_gyeonggi]

    # Tier 2: Districts
    # 서울 산하: 강남구, 서초구
    pos_gangnam = (-2.8, 0.5)
    pos_seocho = (-1.2, 0.5)

    # 경기 산하: 수원시, 성남시
    pos_suwon = (1.2, 0.5)
    pos_seongnam = (2.8, 0.5)

    pos_t2 = [pos_gangnam, pos_seocho, pos_suwon, pos_seongnam]
    district_names = ['Gangnam', 'Seocho', 'Suwon', 'Seongnam']

    # 3. Static Elements - Connections (배경)
    # Tier 2 -> Tier 1 connections
    for i in [0, 1]:  # 서울 districts
        ax.plot([pos_t2[i][0], pos_seoul[0]], [pos_t2[i][1], pos_seoul[1]],
                color='#cccccc', lw=1.5, zorder=1, alpha=0.6)
    for i in [2, 3]:  # 경기 districts
        ax.plot([pos_t2[i][0], pos_gyeonggi[0]], [pos_t2[i][1], pos_gyeonggi[1]],
                color='#cccccc', lw=1.5, zorder=1, alpha=0.6)

    # Tier 1 -> Tier 0 connections
    ax.plot([pos_seoul[0], pos_t0[0]], [pos_seoul[1], pos_t0[1]],
            color='#999999', lw=2.5, zorder=1, alpha=0.7)
    ax.plot([pos_gyeonggi[0], pos_t0[0]], [pos_gyeonggi[1], pos_t0[1]],
            color='#999999', lw=2.5, zorder=1, alpha=0.7)

    # 4. Static Elements - Nodes
    # Tier 설명 레이블 (우측에 배치)
    ax.text(4.2, 4.5, 'Tier 0\nNational', ha='left', va='center',
           fontsize=9, color='#666', style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#999', alpha=0.7))
    ax.text(4.2, 2.5, 'Tier 1\nProvincial', ha='left', va='center',
           fontsize=9, color='#666', style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#999', alpha=0.7))
    ax.text(4.2, 0.5, 'Tier 2\nMunicipal', ha='left', va='center',
           fontsize=9, color='#666', style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#999', alpha=0.7))

    # Tier 2 Nodes (Districts) - 지역 설명 추가
    district_labels = [
        'Gangnam\n(Seoul District)',
        'Seocho\n(Seoul District)',
        'Suwon\n(Gyeonggi City)',
        'Seongnam\n(Gyeonggi City)'
    ]

    for i, (pos, label) in enumerate(zip(pos_t2, district_labels)):
        circle = Circle(pos, 0.25, facecolor='white', edgecolor='#555',
                       linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1]-0.55, label, ha='center', fontsize=9,
               color='#333', fontweight='bold')

    # Tier 1 Nodes (Provinces) - 지역 계층 명시
    # Seoul (Treatment - Blue)
    circle_seoul = Circle(pos_seoul, 0.4, facecolor='#e3f2fd',
                         edgecolor='#1976d2', linewidth=3, zorder=10)
    ax.add_patch(circle_seoul)
    ax.text(pos_seoul[0], pos_seoul[1]+0.05, 'Seoul', ha='center', va='center',
           fontsize=12, fontweight='bold', color='#1565c0', zorder=12)
    ax.text(pos_seoul[0], pos_seoul[1]-0.2, 'Province', ha='center', va='center',
           fontsize=9, style='italic', color='#1565c0', zorder=12)
    ax.text(pos_seoul[0], pos_seoul[1]-0.7, '(Treatment)', ha='center', va='top',
           fontsize=8, color='#1976d2', zorder=12, bbox=dict(boxstyle='round,pad=0.2',
           facecolor='#e3f2fd', edgecolor='#1976d2', alpha=0.7))

    # Gyeonggi (Control - Red)
    circle_gyeonggi = Circle(pos_gyeonggi, 0.4, facecolor='#ffebee',
                            edgecolor='#d32f2f', linewidth=3, zorder=10)
    ax.add_patch(circle_gyeonggi)
    ax.text(pos_gyeonggi[0], pos_gyeonggi[1]+0.05, 'Gyeonggi', ha='center', va='center',
           fontsize=12, fontweight='bold', color='#c62828', zorder=12)
    ax.text(pos_gyeonggi[0], pos_gyeonggi[1]-0.2, 'Province', ha='center', va='center',
           fontsize=9, style='italic', color='#c62828', zorder=12)
    ax.text(pos_gyeonggi[0], pos_gyeonggi[1]-0.7, '(Control)', ha='center', va='top',
           fontsize=8, color='#d32f2f', zorder=12, bbox=dict(boxstyle='round,pad=0.2',
           facecolor='#ffebee', edgecolor='#d32f2f', alpha=0.7))

    # Tier 0 Node (KDCA)
    circle_kdca = Circle(pos_t0, 0.5, facecolor='#1a237e', edgecolor='black',
                        linewidth=3, zorder=10)
    ax.add_patch(circle_kdca)
    ax.text(pos_t0[0], pos_t0[1], 'KDCA\nCentral', ha='center', va='center',
           fontsize=12, color='white', fontweight='bold')

    # Watermark: Simulated Data Notice (화면 상단 우측)
    ax.text(4.5, 5.2, 'DEMONSTRATION\nSIMULATED DATA',
           ha='right', va='top', fontsize=8, color='#999999',
           style='italic', alpha=0.6, zorder=5,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                    edgecolor='#cccccc', alpha=0.4))

    # 5. Animation Elements
    # Data Particles (8개: 각 district당 2개)
    particles = []
    for _ in range(8):
        p, = ax.plot([], [], 'o', markersize=10, zorder=20)
        particles.append(p)

    # Encryption Lock Icons (Tier 1에 표시될 자물쇠) - 실제 이미지 사용
    # 노드 왼쪽 위에 바짝 붙여서 2배 크게
    if lock_img is not None:
        # 자물쇠 이미지를 AnnotationBbox로 추가 (2배 크기)
        imagebox_seoul = OffsetImage(lock_img, zoom=0.6)
        ab_seoul = AnnotationBbox(imagebox_seoul, (pos_seoul[0]-0.5, pos_seoul[1]+0.5),
                                  frameon=False, box_alignment=(1.0, 0.0), zorder=30)
        lock_seoul_img = ax.add_artist(ab_seoul)
        lock_seoul_img.set_visible(False)

        imagebox_gyeonggi = OffsetImage(lock_img, zoom=0.6)
        ab_gyeonggi = AnnotationBbox(imagebox_gyeonggi, (pos_gyeonggi[0]-0.5, pos_gyeonggi[1]+0.5),
                                     frameon=False, box_alignment=(1.0, 0.0), zorder=30)
        lock_gyeonggi_img = ax.add_artist(ab_gyeonggi)
        lock_gyeonggi_img.set_visible(False)
    else:
        lock_seoul_img = None
        lock_gyeonggi_img = None

    # 자물쇠 텍스트 라벨 (Phase 2에서만 잠깐 표시)
    lock_seoul_text = ax.text(pos_seoul[0]+0.3, pos_seoul[1]+0.65, 'ENCRYPTED',
                        fontsize=9, alpha=0, zorder=15, fontweight='bold',
                        color='#ffd700', bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='black', alpha=0, edgecolor='#ffd700'))
    lock_gyeonggi_text = ax.text(pos_gyeonggi[0]+0.3, pos_gyeonggi[1]+0.65, 'ENCRYPTED',
                           fontsize=9, alpha=0, zorder=15, fontweight='bold',
                           color='#ffd700', bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='black', alpha=0, edgecolor='#ffd700'))

    # 6. DiD Result Panel (Inset) - 좌측 상단, 최상위 레이어
    ax_did = ax.inset_axes([0.05, 0.60, 0.40, 0.35])
    ax_did.set_facecolor('white')
    ax_did.patch.set_edgecolor('#333')
    ax_did.patch.set_linewidth(2)
    ax_did.patch.set_alpha(0.98)
    ax_did.set_zorder(100)  # 최상위 레이어
    ax_did.set_visible(False)

    # DiD Graph Elements (최상위 레이어)
    did_line_counterfactual, = ax_did.plot([], [], 'k--', lw=2,
                                          label='Counterfactual', alpha=0.7, zorder=101)
    did_line_treatment, = ax_did.plot([], [], color='#1976d2', lw=3,
                                     label='Seoul (Treated)', zorder=102)
    did_line_control, = ax_did.plot([], [], color='#d32f2f', lw=3,
                                   label='Control group (all provinces)', zorder=103)

    # The inset connects pre/post means; its horizontal axis is not actual time.

    # Result Text
    did_result_text = ax_did.text(0.5, 0.95, '', ha='center', va='top',
                                 transform=ax_did.transAxes, fontsize=12,
                                 fontweight='bold', color='#d32f2f', zorder=105)

    # 7. Phase Text (DiD 그래프보다 낮은 레이어)
    phase_text = ax.text(-4.8, 5.2, '', fontsize=16, fontweight='bold',
                        color='#1a237e', zorder=5)
    sub_text = ax.text(-4.8, 4.9, '', fontsize=11, color='#555', zorder=5)

    # 8. Animation Logic
    frames = 300  # 10 seconds at 30fps

    def update(frame):
        t = frame / frames

        # Phase 1: Data Generation (0 ~ 35%)
        if t < 0.35:
            phase_text.set_text('Phase 1: Data Generation (Tier 2)')
            sub_text.set_text('District-level data encryption and upload')
            phase_text.set_alpha(1)
            sub_text.set_alpha(1)

            ax_did.set_visible(False)
            lock_seoul_text.set_alpha(0)
            lock_gyeonggi_text.set_alpha(0)
            # bbox도 숨김
            lock_seoul_text.get_bbox_patch().set_alpha(0)
            lock_gyeonggi_text.get_bbox_patch().set_alpha(0)
            if lock_seoul_img:
                lock_seoul_img.set_visible(False)
                lock_gyeonggi_img.set_visible(False)

            local_t = t / 0.35
            for i, p in enumerate(particles):
                # 각 district에서 시작
                district_idx = i // 2
                start = pos_t2[district_idx]
                end = pos_seoul if district_idx < 2 else pos_gyeonggi

                curr_x = start[0] + (end[0] - start[0]) * local_t
                curr_y = start[1] + (end[1] - start[1]) * local_t

                p.set_data([curr_x], [curr_y])
                # 서울(파랑), 경기(빨강) Raw Data
                p.set_color('#1976d2' if district_idx < 2 else '#d32f2f')
                p.set_alpha(1.0)
                p.set_markersize(10)

        # Phase 2: Blind Aggregation (35% ~ 50%)
        elif t < 0.50:
            phase_text.set_text('Phase 2: Homomorphic Aggregation (Tier 1)')
            sub_text.set_text('Provinces cannot decrypt - Blind Aggregation only')
            phase_text.set_alpha(1)
            sub_text.set_alpha(1)

            # Particles arrive at Tier 1 and turn golden (encrypted)
            for i, p in enumerate(particles):
                district_idx = i // 2
                target = pos_seoul if district_idx < 2 else pos_gyeonggi
                p.set_data([target[0]], [target[1]])
                p.set_color('#ffd700')  # Gold (Encrypted)
                # Pulsing effect
                pulse = 1.0 + 0.3 * np.sin(frame * 0.3)
                p.set_markersize(12 * pulse)

            # Show lock icons and text (Phase 2만)
            lock_alpha = min(1.0, (t - 0.35) / 0.1)
            lock_seoul_text.set_alpha(lock_alpha)  # ENCRYPTED 텍스트
            lock_gyeonggi_text.set_alpha(lock_alpha)
            # bbox도 함께 페이드인
            lock_seoul_text.get_bbox_patch().set_alpha(lock_alpha * 0.8)
            lock_gyeonggi_text.get_bbox_patch().set_alpha(lock_alpha * 0.8)
            if lock_seoul_img and lock_alpha > 0.1:
                lock_seoul_img.set_visible(True)  # 자물쇠 이미지
                lock_gyeonggi_img.set_visible(True)

        # Phase 3: Central Transmission (50% ~ 70%)
        elif t < 0.70:
            phase_text.set_text('Phase 3: Central Transmission (Tier 1 → Tier 0)')
            sub_text.set_text('Encrypted aggregated data sent to central government')
            phase_text.set_alpha(1)
            sub_text.set_alpha(1)

            local_t = (t - 0.50) / 0.20
            for i, p in enumerate(particles):
                district_idx = i // 2
                start = pos_seoul if district_idx < 2 else pos_gyeonggi
                end = pos_t0

                curr_x = start[0] + (end[0] - start[0]) * local_t
                curr_y = start[1] + (end[1] - start[1]) * local_t

                p.set_data([curr_x], [curr_y])
                p.set_color('#ffd700')  # Maintain encryption color
                p.set_markersize(10)

            # Phase 3: 텍스트는 페이드아웃, 자물쇠 이미지만 유지
            text_fade = max(0, 1.0 - (t - 0.50) / 0.05)  # 빠르게 사라짐
            lock_seoul_text.set_alpha(text_fade)
            lock_gyeonggi_text.set_alpha(text_fade)
            # bbox도 함께 페이드아웃
            lock_seoul_text.get_bbox_patch().set_alpha(text_fade * 0.8)
            lock_gyeonggi_text.get_bbox_patch().set_alpha(text_fade * 0.8)
            if lock_seoul_img:
                lock_seoul_img.set_visible(True)  # 자물쇠는 계속 표시
                lock_gyeonggi_img.set_visible(True)

        # Phase 4: DiD Analysis (70% ~ 100%)
        else:
            # Phase 4에서는 DiD 그래프에 집중하도록 phase 텍스트 숨김
            phase_text.set_alpha(0)
            sub_text.set_alpha(0)

            # Hide particles
            for p in particles:
                p.set_alpha(0)

            # Phase 4: 텍스트는 이미 숨김, 자물쇠만 페이드아웃
            lock_seoul_text.set_alpha(0)  # 텍스트는 완전히 숨김
            lock_gyeonggi_text.set_alpha(0)
            # bbox도 완전히 숨김
            lock_seoul_text.get_bbox_patch().set_alpha(0)
            lock_gyeonggi_text.get_bbox_patch().set_alpha(0)

            # 자물쇠 이미지는 Phase 4 시작 직후 사라짐
            if lock_seoul_img and (t - 0.70) > 0.05:
                lock_seoul_img.set_visible(False)
                lock_gyeonggi_img.set_visible(False)

            # Show DiD graph
            ax_did.set_visible(True)
            local_t = min(1.0, (t - 0.70) / 0.25)

            # Schematic interpolation of the measured group-period means, not daily data.
            x = np.linspace(0, 10, 100)
            y_control = control_pre + (control_post-control_pre)*(x/10)
            y_counterfactual = treated_pre + (control_post-control_pre)*(x/10)
            y_treatment = treated_pre + (treated_post-treated_pre)*(x/10)

            # Animate graph drawing
            limit = int(len(x) * local_t)
            did_line_control.set_data(x[:limit], y_control[:limit])
            did_line_counterfactual.set_data(x[:limit], y_counterfactual[:limit])
            did_line_treatment.set_data(x[:limit], y_treatment[:limit])

            ax_did.set_xlim(0, 10)
            lower = min(control_pre, control_post, treated_pre, treated_post)
            upper = max(control_pre, control_post, treated_pre, treated_post, y_counterfactual[-1])
            ax_did.set_ylim(lower-.2, upper+.2)
            ax_did.set_xlabel('Pre to post (schematic)', fontsize=9)
            ax_did.set_ylabel('Mean log(1 + observed cases)', fontsize=9)
            ax_did.set_title('Research contrast; diagram shows selected nodes',
                           fontsize=11, fontweight='bold', pad=10)
            ax_did.legend(fontsize=8, loc='upper left')
            ax_did.grid(True, alpha=0.3, linestyle='--')

            # Show result text
            if local_t > 0.7:
                inference = result['inference']
                status = 'Inference withheld' if inference['status'] == 'withheld' else f"Exploratory p={inference['p_value']:.3g}"
                did_result_text.set_text(f"Contrast: {result['effect']:+.3f} log units\n{status}")

        # Return 리스트 (자물쇠 이미지는 Artist이므로 별도 처리)
        return_list = particles + [phase_text, sub_text, lock_seoul_text, lock_gyeonggi_text,
                                   did_line_counterfactual, did_line_treatment,
                                   did_line_control, did_result_text]
        return return_list

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=frames,
                                 interval=33, blit=True)

    # Save
    output_path = str(root / 'results/policy_analysis_current.mp4')
    ani.save(output_path, fps=30, codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p'],
            metadata={'title': '3-Tier Hierarchical F-Causal Analysis',
                     'artist': 'F-Causal Research Team'})

    print(f"\n{'='*70}")
    print(f"✅ Hierarchical Visualization Video Generated!")
    print(f"{'='*70}")
    print(f"   File: {output_path}")
    print(f"   Resolution: 1920x1080 (16:9)")
    print(f"   Duration: 10 seconds (300 frames @ 30fps)")
    print(f"   Content:")
    print(f"     - Tier 2: Gangnam, Seocho, Suwon, Seongnam")
    print(f"     - Tier 1: Seoul (Treatment), Gyeonggi (Control)")
    print(f"     - Tier 0: KDCA Central Government")
    print(f"     - Research contrast: {result['effect']:+.6f} log1p units")
    print(f"     - Inference: {result['inference']['status']}")
    print(f"{'='*70}\n")

    plt.close()

if __name__ == "__main__":
    create_demo1_hierarchical()
