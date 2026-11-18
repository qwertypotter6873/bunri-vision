/* Bunri-Vision Browser On-Device AI Engine (ONNX Runtime Web WASM)
 *
 * Mirrors the exact 4-stage domain preprocessing, input validation guardrails,
 * single-pass 7x7 Spatial CAM visualization, and Korean waste-sorting decision
 * engine from src/preprocessing.py, app/inference.py, and src/domain_rules.py
 * so the service runs 100% serverlessly on GitHub Pages with zero external APIs.
 */
(() => {
  const CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
  ];

  const RECYCLABLE_SET = new Set([
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Textile Trash",
  ]);

  const KOREAN_WASTE_RULES = {
    Cardboard: {
      en_name: "Cardboard",
      ko_name: "골판지·택배 상자",
      bin_category: "종이·골판지류 전용 수거함",
      bin_badge_type: "recyclable",
      recyclable_status: "재활용 가능 (이물질 제거 필수)",
      steps: [
        "상자에 붙은 박스테이프, 운송장 스티커(감열지), 철핀을 완전히 제거합니다.",
        "부피를 줄이기 위해 상자를 납작하게 펼쳐 접어 묶음 배출합니다.",
        "기름·음식물 국물이 배어든 피자박스/치킨박스 속지는 떼어내 일반 종량제 봉투에 버립니다.",
      ],
      penalty_trap:
        "운송장 스티커와 비닐 코팅 테이프를 떼지 않고 배출하면 제지 공정 불량 및 수거 거부 사유가 됩니다.",
      examples: "택배 박스, 골판지 포장재, 계란 종이판(오염 없는 것), 골판지 완충재",
    },
    "Food Organics": {
      en_name: "Food Organics",
      ko_name: "음식물류 폐기물",
      bin_category: "음식물 전용 종량제 봉투 / RFID 수거함",
      bin_badge_type: "food",
      recyclable_status: "음식물 사료·퇴비화 대상",
      steps: [
        "물기를 최대한 짜서 수분을 제거한 뒤 음식물 전용 수거함·봉투에 배출합니다.",
        "비닐, 이쑤시개, 병뚜껑, 일회용 나무젓가락 등 이물질이 섞이지 않도록 분리합니다.",
        "염분이 많은 국물·장류는 건더기만 걸러서 배출합니다.",
      ],
      penalty_trap:
        "동물이 먹을 수 없는 딱딱한 뼈다귀, 조개·달걀 껍데기, 과일 씨앗(복숭아·아보카도 씨)은 음식물이 아닌 일반 종량제로 배출해야 합니다.",
      examples: "남은 밥·반찬, 과일 껍질(부드러운 것), 채소 자투리, 빵·면류 잔여물",
    },
    Glass: {
      en_name: "Glass",
      ko_name: "유리병·유리 용기",
      bin_category: "유리병류 수거함 (깨진 유리는 불연성 마대)",
      bin_badge_type: "recyclable",
      recyclable_status: "재활용 가능 (내열유리·파손유리 제외)",
      steps: [
        "내용물을 깨끗이 비우고 물로 헹군 뒤 금속·플라스틱 뚜껑을 분리합니다.",
        "담배꽁초나 이물질을 병 안에 넣지 않고 깨지지 않게 수거함에 넣습니다.",
        "소주병·맥주병 등 빈용기보증금 대상 병은 편의점·마트에 반납해 보증금을 환급받으세요.",
      ],
      penalty_trap:
        "깨진 유리, 도자기·사기그릇, 내열유리(전자레인지용 용기), 거울은 절대 유리병 수거함에 넣으면 안 되며 신문지에 싸서 종량제 또는 불연성 마대로 배출해야 합니다.",
      examples: "음료수 병, 맥주·소주병, 잼·소스 유리병, 화장품 유리 용기",
    },
    Metal: {
      en_name: "Metal",
      ko_name: "금속캔·고철류",
      bin_category: "캔·고철류 전용 수거함",
      bin_badge_type: "recyclable",
      recyclable_status: "재활용 우수 품목",
      steps: [
        "음료·통조림 캔은 내용물을 비우고 물로 헹군 뒤 가능하면 발로 밟아 압착합니다.",
        "부탄가스·살충제·헤어스프레이 등 압력 용기는 통풍이 잘되는 야외에서 구멍을 뚫어 잔여 가스를 완전히 제거합니다.",
        "플라스틱 뚜껑이나 고무 패킹이 결합된 경우 분리하여 각각 배출합니다.",
      ],
      penalty_trap:
        "잔여 가스가 남은 부탄가스·스프레이 캔을 구멍 뚫지 않고 배출하면 압축 수거 차량 화재·폭발 사고의 직접 원인이 됩니다.",
      examples: "알루미늄 음료캔, 통조림캔, 부탄가스캔, 알루미늄 호일(깨끗한 것), 공구·철사 고철",
    },
    "Miscellaneous Trash": {
      en_name: "Miscellaneous Trash",
      ko_name: "복합재질·오염 일반쓰레기",
      bin_category: "일반 종량제 봉투",
      bin_badge_type: "general",
      recyclable_status: "재활용 불가 (소각·매립 대상)",
      steps: [
        "음식물·기름으로 심하게 오염되어 세척이 불가능한 비닐·종이·용기는 일반 종량제 봉투에 넣습니다.",
        "여러 재질이 강하게 접착된 복합소재(영수증 감열지, 보온·보냉팩, 칫솔, 고무장갑 등)도 종량제 봉투로 배출합니다.",
        "날카로운 조각이 포함된 경우 종이로 감싸 환경미화원 부상을 예방합니다.",
      ],
      penalty_trap:
        "오염된 일회용기나 영수증·물티슈를 재활용함에 섞어 버리면 전체 재활용 수거물이 오염되어 과태료(최대 10~30만 원) 부과 대상이 됩니다.",
      examples: "오염된 포장재, 영수증(감열지), 사용한 휴지·물티슈, 빨대, 기저귀, 복합재질 쓰레기",
    },
    Paper: {
      en_name: "Paper",
      ko_name: "일반 종이·인쇄물",
      bin_category: "종이류 수거함 (종이팩은 별도 배출)",
      bin_badge_type: "recyclable",
      recyclable_status: "재활용 가능 (비닐코팅·감열지 제외)",
      steps: [
        "물기나 음식물이 묻지 않도록 하고 구겨지지 않게 펴서 차곡차곡 묶어 배출합니다.",
        "노트·책자의 스프링, 비닐 표지, 스테이플러 심을 제거한 뒤 배출합니다.",
        "우유팩·두유팩(살균팩/멸균팩)은 일반 폐지와 펄프 재질이 다르므로 헹궈서 펼쳐 말린 뒤 종이팩 전용 수거함에 배출합니다.",
      ],
      penalty_trap:
        "영수증(감열지), 금박·은박지, 비닐 코팅 전단지, 오염된 핸드타월·휴지는 종이류가 아닌 일반 종량제 봉투에 버려야 합니다.",
      examples: "신문지, 복사용지, 책·잡지, 노트, 종이 쇼핑백(코팅·끈 제거)",
    },
    Plastic: {
      en_name: "Plastic",
      ko_name: "플라스틱 용기·페트류",
      bin_category: "투명페트병 전용함 / 플라스틱류 수거함",
      bin_badge_type: "recyclable",
      recyclable_status: "재활용 가능 (비우기·헹구기·비닐제거)",
      steps: [
        "생수·음료 투명 페트병은 내용물을 비우고 비닐 라벨을 떼어낸 뒤 압착하여 '투명페트병 전용 수거함'에 별도 배출합니다.",
        "유색 플라스틱·배달용기는 음식물·양념을 깨끗이 씻어낸 뒤 일반 플라스틱 수거함에 넣습니다.",
        "샴푸·세제 용기의 펌프(금속 스프링 포함)는 분리하여 일반쓰레기로 버리고 본체만 배출합니다.",
      ],
      penalty_trap:
        "투명 페트병을 일반 유색 플라스틱과 혼합 배출하거나, 씻지 않은 마라탕·떡볶이 빨간 기름 자국 용기를 그대로 배출하면 재활용 선별에서 탈락합니다.",
      examples: "투명 생수·음료 페트병, 세제·샴푸 통, 플라스틱 배달용기, 테이크아웃 컵",
    },
    "Textile Trash": {
      en_name: "Textile Trash",
      ko_name: "폐의류·섬유류",
      bin_category: "의류수거함 (오염·솜류는 종량제/대형폐기물)",
      bin_badge_type: "special",
      recyclable_status: "상태에 따라 조건부 재활용",
      steps: [
        "세탁되어 깨끗하고 다시 입을 수 있는 옷·가방·신발(짝 맞음)은 건조한 상태로 의류수거함에 넣습니다.",
        "젖었거나 기름·페인트·곰팡이로 오염된 옷, 걸레, 양말·속옷류는 의류수거함이 아닌 일반 종량제 봉투에 배출합니다.",
        "솜이불, 베개, 방석, 캐리어, 바퀴 달린 신발은 의류수거함 금지 품목이므로 종량제 봉투 또는 대형폐기물 스티커를 부착해 배출합니다.",
      ],
      penalty_trap:
        "젖은 수건·걸레·솜이불을 의류수거함에 넣으면 수거함 전체 의류에 곰팡이가 번져 전량 폐기되며 지자체 단속 대상이 됩니다.",
      examples: "헌 옷, 직물 조각, 가방·모자(재사용 가능 시 의류수거함 / 훼손·오염 시 종량제)",
    },
    Vegetation: {
      en_name: "Vegetation",
      ko_name: "초목류·식물 잔재물",
      bin_category: "일반 종량제 봉투 (음식물쓰레기 절대 금지!)",
      bin_badge_type: "general",
      recyclable_status: "음식물 사료화 불가 → 일반쓰레기",
      steps: [
        "화분에서 죽은 식물, 나뭇가지, 낙엽, 생화 꽃다발은 흙을 털어낸 뒤 일반 종량제 봉투에 버립니다.",
        "부피가 큰 나뭇가지는 30cm 이하로 잘라 묶거나 대형·초목류 전용 마대로 배출합니다.",
        "양파·마늘·옥수수 마른 겉껍질, 대파·쪽파 뿌리 역시 섬유질이 질겨 가축 사료화가 불가능하므로 일반 종량제 봉투에 배출합니다.",
      ],
      penalty_trap:
        "[자취생 과태료 1위 주의] 식물 잎·줄기·뿌리·낙엽을 '유기물이니까 음식물쓰레기겠지'라고 착각해 음식물 수거함에 버리면 분쇄기 고장 및 과태료 부과 대상입니다!",
      examples: "화분 식물·줄기·뿌리, 낙엽, 잔디·잡초, 시든 꽃다발, 마른 채소 껍질·뿌리",
    },
  };

  const ALLOWED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".avif"]);
  const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
  const MIN_IMAGE_DIM = 48;
  const IMAGENET_MEAN = [0.485, 0.456, 0.406];
  const IMAGENET_STD = [0.229, 0.224, 0.225];
  const MODEL_CACHE_NAME = "bunri-vision-onnx-cache-v1";
  const ORT_WASM_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

  // Precompute sRGB [0..255] -> linear RGB LUT for fast CIE-LAB conversion
  const SRGB_TO_LINEAR = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255.0;
    SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }

  class InputValidationError extends Error {
    constructor(code, message) {
      super(message);
      this.code = code;
      this.message = message;
    }
  }

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function computeImageStats(rgb, pixelCount) {
    let lumSum = 0;
    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    const lums = new Float32Array(pixelCount);

    for (let i = 0, p = 0; i < pixelCount; i++, p += 3) {
      const r = rgb[p];
      const g = rgb[p + 1];
      const b = rgb[p + 2];
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      lums[i] = lum;
      lumSum += lum;
      rSum += r;
      gSum += g;
      bSum += b;
    }

    const lumMean = lumSum / pixelCount;
    let varSum = 0;
    for (let i = 0; i < pixelCount; i++) {
      const d = lums[i] - lumMean;
      varSum += d * d;
    }
    const contrastStd = Math.sqrt(varSum / pixelCount);

    const rm = rSum / pixelCount;
    const gm = gSum / pixelCount;
    const bm = bSum / pixelCount;
    const meanAll = (rm + gm + bm) / 3.0 + 1e-6;
    const nr = rm / meanAll;
    const ng = gm / meanAll;
    const nb = bm / meanAll;
    const nMean = (nr + ng + nb) / 3.0;
    const castVar = ((nr - nMean) ** 2 + (ng - nMean) ** 2 + (nb - nMean) ** 2) / 3.0;
    const colorCastIndex = Math.sqrt(castVar);

    return { lumMean, contrastStd, colorCastIndex };
  }

  function shadesOfGrayWhiteBalance(rgb, pixelCount, power = 6.0) {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let i = 0, p = 0; i < pixelCount; i++, p += 3) {
      sumR += Math.pow(rgb[p] / 255.0 + 1e-6, power);
      sumG += Math.pow(rgb[p + 1] / 255.0 + 1e-6, power);
      sumB += Math.pow(rgb[p + 2] / 255.0 + 1e-6, power);
    }
    const normR = Math.pow(sumR / pixelCount, 1.0 / power);
    const normG = Math.pow(sumG / pixelCount, 1.0 / power);
    const normB = Math.pow(sumB / pixelCount, 1.0 / power);
    const target = (normR + normG + normB) / 3.0;

    const sR = clamp(target / (normR + 1e-6), 0.82, 1.22);
    const sG = clamp(target / (normG + 1e-6), 0.82, 1.22);
    const sB = clamp(target / (normB + 1e-6), 0.82, 1.22);

    const out = new Uint8Array(pixelCount * 3);
    for (let i = 0, p = 0; i < pixelCount; i++, p += 3) {
      out[p] = Math.round(clamp((rgb[p] / 255.0) * sR, 0.0, 1.0) * 255.0);
      out[p + 1] = Math.round(clamp((rgb[p + 1] / 255.0) * sG, 0.0, 1.0) * 255.0);
      out[p + 2] = Math.round(clamp((rgb[p + 2] / 255.0) * sB, 0.0, 1.0) * 255.0);
    }
    return out;
  }

  function adaptiveLabClahe(rgb, width, height, clipLimit = 1.8, gridX = 8, gridY = 8, blendAlpha = 0.65) {
    const pixelCount = width * height;
    const lU8 = new Uint8Array(pixelCount);
    const aStar = new Float32Array(pixelCount);
    const bStar = new Float32Array(pixelCount);

    // D65 reference white matching OpenCV RGB2LAB
    const Xn = 0.950456;
    const Yn = 1.0;
    const Zn = 1.088754;

    for (let i = 0, p = 0; i < pixelCount; i++, p += 3) {
      const rl = SRGB_TO_LINEAR[rgb[p]];
      const gl = SRGB_TO_LINEAR[rgb[p + 1]];
      const bl = SRGB_TO_LINEAR[rgb[p + 2]];

      const X = (0.412453 * rl + 0.35758 * gl + 0.180423 * bl) / Xn;
      const Y = (0.212671 * rl + 0.71516 * gl + 0.072169 * bl) / Yn;
      const Z = (0.019334 * rl + 0.119193 * gl + 0.950227 * bl) / Zn;

      const fX = X > 0.008856 ? Math.cbrt(X) : 7.787 * X + 16.0 / 116.0;
      const fY = Y > 0.008856 ? Math.cbrt(Y) : 7.787 * Y + 16.0 / 116.0;
      const fZ = Z > 0.008856 ? Math.cbrt(Z) : 7.787 * Z + 16.0 / 116.0;

      const L = Y > 0.008856 ? 116.0 * fY - 16.0 : 903.3 * Y;
      lU8[i] = clamp(Math.round((L * 255.0) / 100.0), 0, 255);
      aStar[i] = 500.0 * (fX - fY);
      bStar[i] = 200.0 * (fY - fZ);
    }

    // Build 8x8 tile clipped CDF LUTs
    const luts = new Uint8Array(gridY * gridX * 256);
    const hist = new Int32Array(256);

    for (let ty = 0; ty < gridY; ty++) {
      const y0 = Math.floor((ty * height) / gridY);
      const y1 = Math.floor(((ty + 1) * height) / gridY);
      for (let tx = 0; tx < gridX; tx++) {
        const x0 = Math.floor((tx * width) / gridX);
        const x1 = Math.floor(((tx + 1) * width) / gridX);
        hist.fill(0);
        let tilePixels = 0;
        for (let y = y0; y < y1; y++) {
          const rowOff = y * width;
          for (let x = x0; x < x1; x++) {
            hist[lU8[rowOff + x]]++;
            tilePixels++;
          }
        }

        const limit = Math.max(1, Math.floor((clipLimit * tilePixels) / 256.0));
        let clipped = 0;
        for (let v = 0; v < 256; v++) {
          if (hist[v] > limit) {
            clipped += hist[v] - limit;
            hist[v] = limit;
          }
        }
        const redistBatch = Math.floor(clipped / 256);
        const residual = clipped - redistBatch * 256;
        for (let v = 0; v < 256; v++) {
          hist[v] += redistBatch;
        }
        if (residual > 0) {
          const step = Math.max(1, Math.floor(256 / residual));
          for (let v = 0, r = 0; v < 256 && r < residual; v += step, r++) {
            hist[v]++;
          }
        }

        const lutBase = (ty * gridX + tx) * 256;
        let cdf = 0;
        const scale = 255.0 / Math.max(1, tilePixels);
        for (let v = 0; v < 256; v++) {
          cdf += hist[v];
          luts[lutBase + v] = clamp(Math.round(cdf * scale), 0, 255);
        }
      }
    }

    // Bilinear interpolation across 4 neighboring tile LUTs + alpha blend with original L
    const out = new Uint8Array(pixelCount * 3);
    for (let y = 0; y < height; y++) {
      const tyf = ((y + 0.5) * gridY) / height - 0.5;
      const ty0 = clamp(Math.floor(tyf), 0, gridY - 1);
      const ty1 = clamp(ty0 + 1, 0, gridY - 1);
      const wy = clamp(tyf - ty0, 0.0, 1.0);
      const rowOff = y * width;

      for (let x = 0; x < width; x++) {
        const txf = ((x + 0.5) * gridX) / width - 0.5;
        const tx0 = clamp(Math.floor(txf), 0, gridX - 1);
        const tx1 = clamp(tx0 + 1, 0, gridX - 1);
        const wx = clamp(txf - tx0, 0.0, 1.0);

        const idx = rowOff + x;
        const lv = lU8[idx];

        const v00 = luts[(ty0 * gridX + tx0) * 256 + lv];
        const v01 = luts[(ty0 * gridX + tx1) * 256 + lv];
        const v10 = luts[(ty1 * gridX + tx0) * 256 + lv];
        const v11 = luts[(ty1 * gridX + tx1) * 256 + lv];

        const top = v00 * (1.0 - wx) + v01 * wx;
        const bot = v10 * (1.0 - wx) + v11 * wx;
        const lEnh = top * (1.0 - wy) + bot * wy;
        const lBlendU8 = blendAlpha * lEnh + (1.0 - blendAlpha) * lv;

        // Convert CIE-LAB back to sRGB uint8
        const L = (lBlendU8 * 100.0) / 255.0;
        const fY = (L + 16.0) / 116.0;
        const fX = fY + aStar[idx] / 500.0;
        const fZ = fY - bStar[idx] / 200.0;

        const xr = fX > 0.2068966 ? fX * fX * fX : (fX - 16.0 / 116.0) / 7.787;
        const yr = L > 7.9996 ? fY * fY * fY : L / 903.3;
        const zr = fZ > 0.2068966 ? fZ * fZ * fZ : (fZ - 16.0 / 116.0) / 7.787;

        const X = xr * Xn;
        const Y = yr * Yn;
        const Z = zr * Zn;

        let rl = 3.240479 * X - 1.53715 * Y - 0.498535 * Z;
        let gl = -0.969256 * X + 1.875991 * Y + 0.041556 * Z;
        let bl = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;

        rl = rl <= 0.0 ? 0.0 : rl >= 1.0 ? 1.0 : rl;
        gl = gl <= 0.0 ? 0.0 : gl >= 1.0 ? 1.0 : gl;
        bl = bl <= 0.0 ? 0.0 : bl >= 1.0 ? 1.0 : bl;

        const sr = rl <= 0.0031308 ? 12.92 * rl : 1.055 * Math.pow(rl, 1.0 / 2.4) - 0.055;
        const sg = gl <= 0.0031308 ? 12.92 * gl : 1.055 * Math.pow(gl, 1.0 / 2.4) - 0.055;
        const sb = bl <= 0.0031308 ? 12.92 * bl : 1.055 * Math.pow(bl, 1.0 / 2.4) - 0.055;

        const p = idx * 3;
        out[p] = clamp(Math.round(sr * 255.0), 0, 255);
        out[p + 1] = clamp(Math.round(sg * 255.0), 0, 255);
        out[p + 2] = clamp(Math.round(sb * 255.0), 0, 255);
      }
    }
    return out;
  }

  function resizeAreaAndCenterCrop(rgb, srcW, srcH, imageSize = 224, cropRatio = 0.9) {
    const targetOuter = Math.round(imageSize / cropRatio);
    let newW;
    let newH;
    if (srcH < srcW) {
      newH = targetOuter;
      newW = Math.round(srcW * (targetOuter / srcH));
    } else {
      newW = targetOuter;
      newH = Math.round(srcH * (targetOuter / srcW));
    }

    const top = Math.max(0, Math.floor((newH - imageSize) / 2));
    const left = Math.max(0, Math.floor((newW - imageSize) / 2));

    // Exact box/area downsampling (matching cv2.INTER_AREA) directly on the cropped 224x224 window
    const out = new Uint8Array(imageSize * imageSize * 3);
    const scaleX = srcW / newW;
    const scaleY = srcH / newH;

    for (let dy = 0; dy < imageSize; dy++) {
      const sy0 = (dy + top) * scaleY;
      const sy1 = (dy + top + 1) * scaleY;
      const yStart = Math.floor(sy0);
      const yEnd = Math.min(srcH - 1, Math.ceil(sy1) - 1);

      for (let dx = 0; dx < imageSize; dx++) {
        const sx0 = (dx + left) * scaleX;
        const sx1 = (dx + left + 1) * scaleX;
        const xStart = Math.floor(sx0);
        const xEnd = Math.min(srcW - 1, Math.ceil(sx1) - 1);

        let rAcc = 0;
        let gAcc = 0;
        let bAcc = 0;
        let wAcc = 0;

        for (let sy = yStart; sy <= yEnd; sy++) {
          const wy = Math.min(sy + 1, sy1) - Math.max(sy, sy0);
          if (wy <= 0) continue;
          const rowBase = sy * srcW;
          for (let sx = xStart; sx <= xEnd; sx++) {
            const wx = Math.min(sx + 1, sx1) - Math.max(sx, sx0);
            if (wx <= 0) continue;
            const w = wx * wy;
            const sp = (rowBase + sx) * 3;
            rAcc += rgb[sp] * w;
            gAcc += rgb[sp + 1] * w;
            bAcc += rgb[sp + 2] * w;
            wAcc += w;
          }
        }

        const dp = (dy * imageSize + dx) * 3;
        const invW = wAcc > 0 ? 1.0 / wAcc : 0.0;
        out[dp] = clamp(Math.round(rAcc * invW), 0, 255);
        out[dp + 1] = clamp(Math.round(gAcc * invW), 0, 255);
        out[dp + 2] = clamp(Math.round(bAcc * invW), 0, 255);
      }
    }
    return out;
  }

  function toChwNormalizedTensor(rgb224, imageSize = 224) {
    const plane = imageSize * imageSize;
    const tensor = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < plane; i++, p += 3) {
      tensor[i] = (rgb224[p] / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      tensor[plane + i] = (rgb224[p + 1] / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      tensor[2 * plane + i] = (rgb224[p + 2] / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
    return tensor;
  }

  function cubicWeight(t) {
    const at = Math.abs(t);
    const a = -0.75; // OpenCV INTER_CUBIC Catmull-Rom alpha = -0.75
    if (at <= 1.0) {
      return (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0;
    }
    if (at < 2.0) {
      return a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a;
    }
    return 0.0;
  }

  function renderCamOverlay(rgb224, cam7x7, alpha = 0.44) {
    const kSize = 7;
    const camPos = new Float32Array(49);
    let minV = Infinity;
    let maxV = -Infinity;
    for (let i = 0; i < 49; i++) {
      const v = Math.max(cam7x7[i], 0.0);
      camPos[i] = v;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const range = maxV - minV;
    if (range > 1e-6) {
      for (let i = 0; i < 49; i++) {
        camPos[i] = (camPos[i] - minV) / range;
      }
    } else {
      camPos.fill(0);
    }

    const outSize = 224;
    const out = new Uint8Array(outSize * outSize * 3);
    const scale = kSize / outSize;

    for (let y = 0; y < outSize; y++) {
      const sy = (y + 0.5) * scale - 0.5;
      const iy = Math.floor(sy);
      const fy = sy - iy;

      for (let x = 0; x < outSize; x++) {
        const sx = (x + 0.5) * scale - 0.5;
        const ix = Math.floor(sx);
        const fx = sx - ix;

        let val = 0.0;
        for (let m = -1; m <= 2; m++) {
          const wy = cubicWeight(m - fy);
          const cy = clamp(iy + m, 0, kSize - 1);
          for (let n = -1; n <= 2; n++) {
            const wx = cubicWeight(fx - n);
            const cx = clamp(ix + n, 0, kSize - 1);
            val += camPos[cy * kSize + cx] * wx * wy;
          }
        }
        val = clamp(val, 0.0, 1.0);

        // Exact OpenCV COLORMAP_JET mapping
        const jetR = clamp(1.5 - Math.abs(4.0 * val - 3.0), 0.0, 1.0) * 255.0;
        const jetG = clamp(1.5 - Math.abs(4.0 * val - 2.0), 0.0, 1.0) * 255.0;
        const jetB = clamp(1.5 - Math.abs(4.0 * val - 1.0), 0.0, 1.0) * 255.0;

        const p = (y * outSize + x) * 3;
        out[p] = clamp(Math.round((1.0 - alpha) * rgb224[p] + alpha * jetR), 0, 255);
        out[p + 1] = clamp(Math.round((1.0 - alpha) * rgb224[p + 1] + alpha * jetG), 0, 255);
        out[p + 2] = clamp(Math.round((1.0 - alpha) * rgb224[p + 2] + alpha * jetB), 0, 255);
      }
    }
    return out;
  }

  function rgbArrayToDataUrl(rgb, width, height, quality = 0.88) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(width, height);
    const dst = imgData.data;
    for (let i = 0, sp = 0, dp = 0; i < width * height; i++, sp += 3, dp += 4) {
      dst[dp] = rgb[sp];
      dst[dp + 1] = rgb[sp + 1];
      dst[dp + 2] = rgb[sp + 2];
      dst[dp + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas.toDataURL("image/jpeg", quality);
  }

  async function validateAndDecodeBlob(blob, filename = "upload.jpg", contentType = "") {
    if (!blob || blob.size === 0) {
      throw new InputValidationError("EMPTY_FILE", "비어 있는 파일입니다. 폐기물 사진 파일을 선택해 주세요.");
    }
    if (blob.size > MAX_UPLOAD_BYTES) {
      const sizeMb = (blob.size / (1024 * 1024)).toFixed(1);
      throw new InputValidationError(
        "FILE_TOO_LARGE",
        `파일 크기(${sizeMb}MB)가 허용 한도(15MB)를 초과했습니다. 용량을 줄여 다시 업로드해 주세요.`
      );
    }

    const lowerName = (filename || "").toLowerCase();
    if (lowerName.includes(".")) {
      const ext = "." + lowerName.split(".").pop();
      if (!ALLOWED_EXTENSIONS.has(ext)) {
        throw new InputValidationError(
          "UNSUPPORTED_EXTENSION",
          `지원하지 않는 파일 형식('${ext}')입니다. JPG, PNG, WEBP, BMP 이미지 파일만 판독할 수 있습니다.`
        );
      }
    }

    const mime = contentType || blob.type || "";
    if (mime && !mime.startsWith("image/") && mime !== "application/octet-stream") {
      throw new InputValidationError(
        "UNSUPPORTED_MIME",
        `이미지 파일이 아닙니다(감지된 타입: ${mime}). 폐기물 사진을 업로드해 주세요.`
      );
    }

    let bitmap;
    try {
      bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
    } catch {
      throw new InputValidationError(
        "CORRUPTED_IMAGE",
        "손상되었거나 읽을 수 없는 이미지 데이터입니다. 정상적인 사진 파일인지 확인해 주세요."
      );
    }

    const origW = bitmap.width;
    const origH = bitmap.height;
    if (origW < MIN_IMAGE_DIM || origH < MIN_IMAGE_DIM) {
      bitmap.close();
      throw new InputValidationError(
        "IMAGE_TOO_SMALL",
        `이미지 해상도(${origW}×${origH}px)가 너무 낮아 재질 질감을 판별할 수 없습니다. 최소 ${MIN_IMAGE_DIM}×${MIN_IMAGE_DIM}px 이상의 사진을 입력해 주세요.`
      );
    }

    // Cap working resolution at short-side <= 768px for crisp speed on 12MP mobile photos;
    // keeps 524x524 RealWaste samples at exact native 524x524 resolution.
    let workW = origW;
    let workH = origH;
    const minSide = Math.min(origW, origH);
    if (minSide > 768) {
      const ratio = 768 / minSide;
      workW = Math.round(origW * ratio);
      workH = Math.round(origH * ratio);
    }

    const canvas = document.createElement("canvas");
    canvas.width = workW;
    canvas.height = workH;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, workW, workH);

    const previewCanvas = document.createElement("canvas");
    previewCanvas.width = 224;
    previewCanvas.height = 224;
    const pCtx = previewCanvas.getContext("2d");
    pCtx.drawImage(bitmap, 0, 0, 224, 224);
    const rawPreviewUrl = previewCanvas.toDataURL("image/jpeg", 0.88);
    bitmap.close();

    const rgba = ctx.getImageData(0, 0, workW, workH).data;
    const pixelCount = workW * workH;
    const rgb = new Uint8Array(pixelCount * 3);
    for (let i = 0, sp = 0, dp = 0; i < pixelCount; i++, sp += 4, dp += 3) {
      rgb[dp] = rgba[sp];
      rgb[dp + 1] = rgba[sp + 1];
      rgb[dp + 2] = rgba[sp + 2];
    }

    const beforeStats = computeImageStats(rgb, pixelCount);
    if (beforeStats.contrastStd < 3.0) {
      throw new InputValidationError(
        "BLANK_OR_SOLID_IMAGE",
        "거의 단색이거나 피사체가 식별되지 않는 빈 화면(표준편차 < 3.0)입니다. 폐기물이 보이도록 촬영해 주세요."
      );
    }

    return { origW, origH, workW, workH, rgb, beforeStats, rawPreviewUrl };
  }

  function buildDecisionAdvice(probsByClass) {
    const ranked = Object.entries(probsByClass).sort((a, b) => b[1] - a[1]);
    const [top1Cls, top1Prob] = ranked[0];
    const [top2Cls, top2Prob] = ranked[1];
    const margin = top1Prob - top2Prob;

    const eps = 1e-12;
    let entropy = 0.0;
    for (const [, p] of ranked) {
      entropy -= p * Math.log(Math.max(p, eps));
    }
    const maxEntropy = Math.log(ranked.length);
    const normEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0.0;

    const rule = KOREAN_WASTE_RULES[top1Cls];
    const alerts = [];

    const isUncertain = top1Prob < 0.42 || normEntropy > 0.74;
    if (isUncertain) {
      alerts.push({
        level: "warning",
        title: "낮은 판독 신뢰도 (재촬영 권장)",
        message:
          `최고 확률이 ${(top1Prob * 100).toFixed(1)}%(정규화 엔트로피 ${normEntropy.toFixed(2)})로 낮습니다. ` +
          "여러 폐기물이 겹쳐 있거나 학습 도메인 외 물체일 수 있으니 단일 품목을 밝은 조명 아래 중앙에 두고 촬영해 주세요.",
      });
    }

    const contaminationProb =
      (probsByClass["Miscellaneous Trash"] || 0.0) + (probsByClass["Food Organics"] || 0.0);
    const contaminationSuspected = RECYCLABLE_SET.has(top1Cls) && contaminationProb >= 0.18;
    if (contaminationSuspected) {
      alerts.push({
        level: "caution",
        title: "표면 오염·이물질 혼입 주의 감지",
        message:
          `주 재질은 '${rule.ko_name}'(${(top1Prob * 100).toFixed(1)}%)이나 오염·음식물 연관 클래스 확률 합계가 ` +
          `${(contaminationProb * 100).toFixed(1)}%로 동반 감지되었습니다. 물 세척으로 이물질이 지워지지 않으면 ` +
          "'일반 종량제 봉투'로 배출하세요.",
      });
    }

    if (margin < 0.2 && !isUncertain) {
      const secondRule = KOREAN_WASTE_RULES[top2Cls];
      alerts.push({
        level: "info",
        title: `유사 품목 경계 확인 (${rule.ko_name} vs ${secondRule.ko_name})`,
        message:
          `1순위 '${rule.ko_name}'(${(top1Prob * 100).toFixed(1)}%)와 2순위 '${secondRule.ko_name}'(${(top2Prob * 100).toFixed(1)}%)의 ` +
          `확률 격차(${(margin * 100).toFixed(1)}%p)가 좁습니다. 두 품목의 배출 기준을 함께 확인하세요.`,
      });
    }

    if (top1Cls === "Vegetation") {
      alerts.push({
        level: "critical",
        title: "음식물쓰레기 오배출 주의 (초목류)",
        message:
          "식물 잎·줄기·화분 잔재·뿌리는 가축 사료화가 불가하여 음식물 수거함 배출 시 과태료 대상입니다. 반드시 일반 종량제 봉투에 버리세요!",
      });
    } else if (top1Cls === "Textile Trash") {
      alerts.push({
        level: "caution",
        title: "건조·오염 여부 확인 필수 (섬유류)",
        message:
          "깨끗하고 마른 의류만 의류수거함에 배출 가능하며, 젖거나 오염된 천·걸레·솜이불은 일반 종량제 봉투(또는 대형폐기물)로 배출해야 합니다.",
      });
    }

    return {
      predicted_class: top1Cls,
      predicted_ko_name: rule.ko_name,
      confidence: Number(top1Prob.toFixed(4)),
      margin_top1_top2: Number(margin.toFixed(4)),
      normalized_entropy: Number(normEntropy.toFixed(4)),
      contamination_score: Number(contaminationProb.toFixed(4)),
      is_uncertain: isUncertain,
      contamination_suspected: contaminationSuspected,
      rule,
      alerts,
      top_predictions: ranked.map(([clsName, prob]) => ({
        class_en: clsName,
        class_ko: KOREAN_WASTE_RULES[clsName].ko_name,
        bin_category: KOREAN_WASTE_RULES[clsName].bin_category,
        probability: Number(prob.toFixed(4)),
      })),
    };
  }

  async function fetchModelWithProgress(modelUrl, onProgress) {
    let cache = null;
    if ("caches" in window) {
      try {
        cache = await caches.open(MODEL_CACHE_NAME);
        const cachedResp = await cache.match(modelUrl);
        if (cachedResp && cachedResp.ok) {
          if (onProgress) onProgress(100, true);
          return await cachedResp.arrayBuffer();
        }
      } catch {
        cache = null;
      }
    }

    const resp = await fetch(modelUrl);
    if (!resp.ok) {
      throw new Error(`모델 가중치 로드 실패 (HTTP ${resp.status})`);
    }

    const contentLength = Number(resp.headers.get("Content-Length") || 33869000);
    if (!resp.body || !resp.body.getReader) {
      const buf = await resp.arrayBuffer();
      if (cache) {
        try {
          await cache.put(modelUrl, new Response(buf.slice(0)));
        } catch {}
      }
      return buf;
    }

    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      if (onProgress && contentLength > 0) {
        const pct = Math.min(99, Math.round((received / contentLength) * 100));
        onProgress(pct, false);
      }
    }

    const full = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      full.set(chunk, offset);
      offset += chunk.byteLength;
    }

    if (cache) {
      try {
        await cache.put(
          modelUrl,
          new Response(full.buffer.slice(0), {
            headers: { "Content-Type": "application/octet-stream" },
          })
        );
      } catch {}
    }
    if (onProgress) onProgress(100, false);
    return full.buffer;
  }

  function makeSolidPngBlob(w, h, r, g, b, mime = "image/png") {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, 0, w, h);
    return new Promise((resolve) => c.toBlob(resolve, mime, 0.92));
  }

  class BunriBrowserEngine {
    constructor(modelUrl = "models/bunri_convnext_tiny_int8.onnx") {
      this.modelUrl = modelUrl;
      this.modelName = "bunri_convnext_tiny_int8.onnx";
      this.session = null;
      this.initPromise = null;
    }

    async init(onProgress) {
      if (this.session) return this.session;
      if (this.initPromise) return this.initPromise;

      this.initPromise = (async () => {
        if (typeof ort === "undefined") {
          throw new Error("ONNX Runtime Web 스크립트를 불러오지 못했습니다.");
        }
        ort.env.wasm.wasmPaths = ORT_WASM_CDN;
        // Single-threaded SIMD WASM avoids SharedArrayBuffer COOP/COEP header requirements on GitHub Pages
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;

        const modelBuffer = await fetchModelWithProgress(this.modelUrl, onProgress);
        const session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });

        // Warmup inference so first user click has predictable latency
        const dummy = new ort.Tensor("float32", new Float32Array(1 * 3 * 224 * 224), [1, 3, 224, 224]);
        await session.run({ input: dummy });
        this.session = session;
        return session;
      })();

      return this.initPromise;
    }

    async predictBlob(blob, filename = "upload.jpg", contentType = "") {
      await this.init();
      const t0 = performance.now();

      const decoded = await validateAndDecodeBlob(blob, filename, contentType);
      const { origW, origH, workW, workH, rgb, beforeStats, rawPreviewUrl } = decoded;
      const pixelCount = workW * workH;

      const wbRgb = shadesOfGrayWhiteBalance(rgb, pixelCount, 6.0);
      const claheRgb = adaptiveLabClahe(wbRgb, workW, workH, 1.8, 8, 8, 0.65);
      const enhanced224 = resizeAreaAndCenterCrop(claheRgb, workW, workH, 224, 0.9);
      const afterStats = computeImageStats(enhanced224, 224 * 224);

      const chwData = toChwNormalizedTensor(enhanced224, 224);
      const inputTensor = new ort.Tensor("float32", chwData, [1, 3, 224, 224]);
      const outputs = await this.session.run({ input: inputTensor });

      const logits = outputs.calibrated_logits.data;
      const camMaps = outputs.cam_maps.data;

      let maxLogit = -Infinity;
      for (let i = 0; i < CLASS_NAMES.length; i++) {
        if (logits[i] > maxLogit) maxLogit = logits[i];
      }
      const exps = new Float64Array(CLASS_NAMES.length);
      let sumExp = 0.0;
      for (let i = 0; i < CLASS_NAMES.length; i++) {
        const e = Math.exp(logits[i] - maxLogit);
        exps[i] = e;
        sumExp += e;
      }

      const probsDict = {};
      let topIdx = 0;
      let topProb = -1;
      for (let i = 0; i < CLASS_NAMES.length; i++) {
        const p = exps[i] / sumExp;
        probsDict[CLASS_NAMES[i]] = p;
        if (p > topProb) {
          topProb = p;
          topIdx = i;
        }
      }

      const camSlice = camMaps.subarray(topIdx * 49, (topIdx + 1) * 49);
      const camOverlay224 = renderCamOverlay(enhanced224, camSlice, 0.44);
      const elapsedMs = Number((performance.now() - t0).toFixed(1));

      const decision = buildDecisionAdvice(probsDict);

      return {
        ...decision,
        inference_ms: elapsedMs,
        model_checkpoint: `${this.modelName} (WASM)`,
        quality_report: {
          width: origW,
          height: origH,
          luminance_mean: Number(beforeStats.lumMean.toFixed(2)),
          contrast_std: Number(beforeStats.contrastStd.toFixed(2)),
          color_cast_index: Number(beforeStats.colorCastIndex.toFixed(4)),
          after_luminance_mean: Number(afterStats.lumMean.toFixed(2)),
          after_contrast_std: Number(afterStats.contrastStd.toFixed(2)),
          after_color_cast_index: Number(afterStats.colorCastIndex.toFixed(4)),
        },
        visuals: {
          raw_image_url: rawPreviewUrl,
          preprocessed_image_url: rgbArrayToDataUrl(enhanced224, 224, 224),
          cam_overlay_url: rgbArrayToDataUrl(camOverlay224, 224, 224),
        },
      };
    }

    async runExceptionTest(caseType) {
      let blob;
      let fname;
      let ctype;
      if (caseType === "non_image_txt") {
        blob = new Blob(["Hello, this is a plain text note, not a waste photo!"], { type: "text/plain" });
        fname = "memo.txt";
        ctype = "text/plain";
      } else if (caseType === "corrupted_binary") {
        const broken = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x42, 0x52, 0x4f, 0x4b, 0x45, 0x4e]);
        blob = new Blob([broken], { type: "image/jpeg" });
        fname = "broken_photo.jpg";
        ctype = "image/jpeg";
      } else if (caseType === "tiny_resolution") {
        blob = await makeSolidPngBlob(24, 24, 120, 160, 90, "image/png");
        fname = "tiny_24x24.png";
        ctype = "image/png";
      } else if (caseType === "blank_solid") {
        blob = await makeSolidPngBlob(256, 256, 240, 240, 240, "image/jpeg");
        fname = "blank_wall.jpg";
        ctype = "image/jpeg";
      } else {
        throw new Error("알 수 없는 예외 테스트 유형입니다.");
      }

      try {
        await validateAndDecodeBlob(blob, fname, ctype);
        return { error: false };
      } catch (err) {
        if (err instanceof InputValidationError) {
          return {
            error: true,
            error_code: err.code,
            tested_file: fname,
            message: err.message,
          };
        }
        throw err;
      }
    }
  }

  window.BunriVision = {
    CLASS_NAMES,
    KOREAN_WASTE_RULES,
    InputValidationError,
    BunriBrowserEngine,
  };
})();
