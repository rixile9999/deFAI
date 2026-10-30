"""
계층형 F-Causal 설정 스크립트

1. 암호화 키 생성 (Public/Secret)
2. 데이터 검증
"""
import os
import sys
import tenseal as ts
from pathlib import Path

def generate_pki():
    """PKI 키 생성 - CKKS 방식"""
    print("🔐 Generating PKI Keys (Public/Secret)...")
    os.makedirs("keys", exist_ok=True)
    
    # CKKS Context 생성
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    
    # Secret Key 포함 (중앙정부용 - 복호화 가능)
    with open("keys/secret.ctx", "wb") as f:
        f.write(context.serialize(save_secret_key=True))
    print("   ✅ Secret key: keys/secret.ctx (Tier 0 only)")
    
    # Secret Key 제거 (지자체용 - 암호화 및 합산만 가능)
    with open("keys/public.ctx", "wb") as f:
        f.write(context.serialize(save_secret_key=False))
    print("   ✅ Public key: keys/public.ctx (Tier 1, 2)")
    
    print("\n⚠️ SECURITY NOTE:")
    print("   - secret.ctx: Only for Tier 0 (Central Government)")
    print("   - public.ctx: For Tier 1 (Province) and Tier 2 (District)")
    print("   - Tier 1 cannot decrypt data, only aggregate!")


def verify_setup():
    """설정 검증"""
    print("\n" + "="*70)
    print("🔍 Verifying Setup")
    print("="*70)
    
    # 키 확인
    if not Path("keys/secret.ctx").exists():
        print("❌ Secret key not found")
        return False
    if not Path("keys/public.ctx").exists():
        print("❌ Public key not found")
        return False
    print("✅ Encryption keys found")
    
    # 데이터 확인
    data_dir = Path("data/districts")
    if not data_dir.exists():
        print("⚠️ District data not found. Run data generator first:")
        print("   python data_gen_hierarchical.py")
        return False
    
    files = list(data_dir.glob("*.xlsx"))
    if len(files) < 3:
        print(f"⚠️ Only {len(files)} district files found. Expected at least 3.")
        return False
    
    print(f"✅ Found {len(files)} district data files")
    
    # 파일 구조 확인
    print("\n📂 Data Structure:")
    provinces = {}
    for f in sorted(files):
        parts = f.stem.split('_')
        if len(parts) >= 2:
            province = parts[0]
            district = parts[1]
            if province not in provinces:
                provinces[province] = []
            provinces[province].append(district)
    
    for prov, dists in provinces.items():
        print(f"   {prov.capitalize():10s}: {', '.join(dists)}")
    
    if 'seoul' not in provinces:
        print("\n⚠️ WARNING: Seoul (treatment group) not found!")
        return False
    
    control_count = sum(1 for p in provinces if p != 'seoul')
    if control_count == 0:
        print("\n⚠️ WARNING: No control group found!")
        return False
    
    print(f"\n✅ Setup verification complete!")
    print(f"   - Treatment: Seoul ({len(provinces.get('seoul', []))} districts)")
    print(f"   - Control: {control_count} provinces")
    
    return True


def main():
    print("\n" + "="*70)
    print("🏛️ F-CAUSAL HIERARCHICAL SETUP")
    print("="*70)
    print("\nThis script prepares the 3-tier federated learning system:")
    print("  Tier 0: Central Government (KDCA) - Secret key holder")
    print("  Tier 1: Provincial offices (Seoul, Gyeonggi, Busan) - Aggregators")
    print("  Tier 2: District offices (Gangnam, Suwon, etc.) - Data holders")
    print()
    
    # 1. 키 생성
    generate_pki()
    
    # 2. 검증
    if verify_setup():
        print("\n" + "="*70)
        print("✅ SETUP COMPLETE - Ready to run hierarchical FL!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Start Tier 0: python run_tier0.py")
        print("  2. Start Tier 1: python run_tier1.py --port 9001 --region seoul")
        print("  3. Start Tier 2: python run_tier2.py --data data/districts/seoul_gangnam.xlsx --target 127.0.0.1:9001")
    else:
        print("\n" + "="*70)
        print("⚠️ SETUP INCOMPLETE")
        print("="*70)
        print("\nPlease run data generator first:")
        print("  python data_gen_hierarchical.py")


if __name__ == "__main__":
    main()
