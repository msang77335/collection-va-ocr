const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Đọc dữ liệu từ file results
const resultsFile = './va-info-list.json';
const imageFolder = './va-frontid-images';

// Python interpreter trong venv
const PYTHON = path.join(__dirname, '.venv', 'bin', 'python3');
const OCR_SCRIPT = path.join(__dirname, 'ocr_card.py');

// Log lines to suppress (same as the grep -v filter in test.md)
const SUPPRESS_PATTERNS = [
  /Model files already/,
  /Creating model/,
  /UserWarning/,
  /warnings\.warn/,
  /cpp_extension/,
];

// Kiểm tra file results tồn tại
if (!fs.existsSync(resultsFile)) {
  console.error(`❌ File không tồn tại: ${resultsFile}`);
  process.exit(1);
}

if (!fs.existsSync(PYTHON)) {
  console.error(`❌ Python venv không tồn tại: ${PYTHON}`);
  console.error(`   Hãy chạy: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`);
  process.exit(1);
}

const resultsFormFile = JSON.parse(fs.readFileSync(resultsFile, 'utf-8'));
const results = resultsFormFile.filter(r => r.filename);

// File lưu kết quả OCR real-time
const outputFileName = `ocr-results_${new Date().toISOString().replace(/:/g, '-').split('.')[0]}.json`;
const ocrResults = [];

// Hàm lưu kết quả real-time
function saveOCRResult(result) {
  ocrResults.push(result);
  fs.writeFileSync(outputFileName, JSON.stringify(ocrResults, null, 2), 'utf-8');
}

// Hàm gọi ocr_card.py trực tiếp (không cần API server)
function processOCR(filename) {
  const imagePath = path.join(imageFolder, filename);

  if (!fs.existsSync(imagePath)) {
    console.log(`  ⚠ File không tồn tại: ${filename}`);
    return null;
  }

  // Temp file để nhận JSON output
  const tempOutput = path.join(os.tmpdir(), `ocr_${Date.now()}_${Math.random().toString(36).slice(2)}.json`);

  try {
    const proc = spawnSync(
      PYTHON,
      [OCR_SCRIPT, '-i', imagePath, '-o', tempOutput],
      { encoding: 'utf-8', timeout: 120000 }
    );

    // In stderr đã lọc bớt noise (giống grep -v trong test.md)
    if (proc.stderr) {
      proc.stderr.split('\n')
        .filter(line => line.trim() && !SUPPRESS_PATTERNS.some(p => p.test(line)))
        .forEach(line => console.log(`  [py] ${line}`));
    }

    if (proc.status !== 0) {
      console.error(`  ✗ ocr_card.py exit ${proc.status}`);
      return null;
    }

    if (!fs.existsSync(tempOutput)) {
      console.error(`  ✗ Không tìm thấy output JSON`);
      return null;
    }

    const raw = JSON.parse(fs.readFileSync(tempOutput, 'utf-8'));

    // Chuẩn hoá về cùng format với API response:
    // { success, side, fields: { ...fields, file, degrees } }
    const { side, file, degrees, ...fields } = raw;
    return {
      success: true,
      side: side || 'unknown',
      fields: { ...fields, file, degrees },
    };

  } catch (err) {
    console.error(`  ✗ Lỗi:`, err.message);
    return null;
  } finally {
    if (fs.existsSync(tempOutput)) fs.unlinkSync(tempOutput);
  }
}

// Hàm chính
function processAllImages() {
  console.log(`Bắt đầu xử lý OCR cho ${results.length} images...\n`);

  for (let i = 0; i < results.length; i++) {
    const { recordId, bankAccountNumber, filename } = results[i];
    console.log(`[${i + 1}/${results.length}] Đang xử lý: ${filename}`);

    const ocrData = processOCR(filename);

    if (ocrData) {
      console.log(`  ✅ OCR thành công  side=${ocrData.side}\n`);
      saveOCRResult({
        recordId,
        bankAccountNumber,
        filename,
        ocrData,
        processedAt: new Date().toISOString(),
        success: true,
      });
    } else {
      console.log(`  ✗ OCR thất bại\n`);
      saveOCRResult({
        recordId,
        bankAccountNumber,
        filename,
        error: 'OCR processing failed',
        processedAt: new Date().toISOString(),
        success: false,
      });
    }
  }

  // Tóm tắt
  const successCount = ocrResults.filter(r => r.success).length;
  const errorCount   = ocrResults.filter(r => !r.success).length;

  console.log(`\n✅ Hoàn thành!`);
  console.log(`\nTổng kết:`);
  console.log(`- Tổng số:    ${results.length}`);
  console.log(`- Thành công: ${successCount}`);
  console.log(`- Lỗi:        ${errorCount}`);
  console.log(`- Kết quả:    ${outputFileName}`);
}

// Chạy script
try {
  processAllImages();
} catch (error) {
  console.error('Lỗi:', error);
  process.exit(1);
}
