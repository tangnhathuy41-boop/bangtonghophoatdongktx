import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Body parser with 30mb limit for high-res bill images
app.use(express.json({ limit: '30mb' }));
app.use(express.urlencoded({ extended: true, limit: '30mb' }));

// Lazy initialization of Gemini client
let geminiClient = null;
function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured');
  }
  if (!geminiClient) {
    geminiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return geminiClient;
}

// API: Quét và nhận diện hóa đơn thông minh bằng AI Gemini
const CANDIDATE_MODELS = ['gemini-3.6-flash', 'gemini-3.8-flash', 'gemini-3.6-pro'];

async function generateContentWithFallback(ai, requestConfig) {
  let lastError = null;
  for (const model of CANDIDATE_MODELS) {
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        console.log(`Đang gọi Gemini AI với model: ${model} (lần thử ${attempt})...`);
        const response = await ai.models.generateContent({
          ...requestConfig,
          model,
        });
        return { response, modelUsed: model };
      } catch (err) {
        lastError = err;
        const msg = (err.message || '').toLowerCase();
        const isTransient =
          msg.includes('503') ||
          msg.includes('unavailable') ||
          msg.includes('high demand') ||
          msg.includes('429') ||
          msg.includes('resource_exhausted') ||
          msg.includes('rate');

        console.warn(`Thử nghiệm model ${model} (lần ${attempt}) không thành công: ${err.message}`);
        if (isTransient && attempt < 2) {
          // Chờ ngắn trước khi thử lại
          await new Promise((r) => setTimeout(r, 1000 * attempt));
          continue;
        }
        break; // Chuyển sang candidate model tiếp theo
      }
    }
  }
  throw lastError;
}

app.post('/api/scan-bill', async (req, res) => {
  try {
    const { imageBase64, mimeType } = req.body;
    if (!imageBase64) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng cung cấp ảnh hóa đơn cần quét.',
      });
    }

    const ai = getGeminiClient();

    // Tách base64 thuần túy nếu có header prefix data:image/...;base64,
    let pureBase64 = imageBase64;
    let detectedMime = mimeType || 'image/jpeg';
    if (imageBase64.includes('base64,')) {
      const parts = imageBase64.split('base64,');
      pureBase64 = parts[1];
      const match = parts[0].match(/data:(.*?);/);
      if (match && match[1]) {
        detectedMime = match[1];
      }
    }

    const prompt = `Bạn là chuyên gia thị giác máy tính AI chuyên đọc và trích xuất dữ liệu hóa đơn thanh toán / phiếu tính tiền (siêu thị WinMart, Co.op, Circle K, GS25, hóa đơn ăn uống, tiền điện nước, nước đóng bình, đồ vệ sinh gia dụng) phục vụ việc chia tiền cho sinh viên phòng KTX tại Việt Nam.

Hãy kiểm tra kỹ hình ảnh và trích xuất:
1. expenseName: Tên ngắn gọn súc tích của khoản chi (VD: "Hóa đơn WinMart (đồ ăn)", "Nước khoáng Lavie 20L", "Nước giặt & Nước lau sàn", "Tiền điện tháng 9", "Hóa đơn Circle K", v.v.).
2. totalAmount: Tổng số tiền thanh toán cuối cùng (dưới dạng số nguyên VNĐ, ví dụ: 86000 hoặc 245000). Hãy tìm dòng TỔNG CỘNG, TỔNG TIỀN, TỔNG THANH TOÁN, TOTAL hoặc số tiền lớn nhất đại diện cho hóa đơn.
3. category: Chọn đúng 1 trong các mục: "Ăn uống", "Sinh hoạt", "Vệ sinh", "Điện nước", hoặc "Khác".
4. merchantName: Tên nhà cung cấp / siêu thị / cửa hàng (VD: WinMart, Circle K, Điện lực, Highlands...).
5. items: Danh sách các món hàng đọc được trên hóa đơn nếu có, gồm tên món (name) và số tiền (price).
6. notes: Ghi chú hữu ích (ví dụ: ngày giờ mua, giảm giá, số lượng món).`;

    const { response, modelUsed } = await generateContentWithFallback(ai, {
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: detectedMime,
              data: pureBase64,
            },
          },
          {
            text: prompt,
          },
        ],
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            expenseName: {
              type: Type.STRING,
              description: 'Tên khoản chi ngắn gọn, rõ ràng',
            },
            totalAmount: {
              type: Type.NUMBER,
              description: 'Tổng tiền thanh toán cuối cùng bằng số nguyên VNĐ',
            },
            category: {
              type: Type.STRING,
              description: 'Phân loại: Ăn uống, Sinh hoạt, Vệ sinh, Điện nước, hoặc Khác',
            },
            merchantName: {
              type: Type.STRING,
              description: 'Tên cửa hàng hoặc thương hiệu',
            },
            items: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  price: { type: Type.NUMBER },
                },
              },
              description: 'Danh sách các mặt hàng trên hóa đơn',
            },
            notes: {
              type: Type.STRING,
              description: 'Ghi chú thêm về ngày giờ hoặc chi tiết hóa đơn',
            },
          },
          required: ['expenseName', 'totalAmount', 'category'],
        },
      },
    });

    console.log(`Đã nhận diện thành công qua model: ${modelUsed}`);
    const rawText = response.text ? response.text.trim() : '{}';
    const result = JSON.parse(rawText);

    return res.json({
      success: true,
      data: {
        expenseName: result.expenseName || 'Khoản chi hóa đơn',
        totalAmount: Math.round(Number(result.totalAmount) || 0),
        category: ['Ăn uống', 'Sinh hoạt', 'Vệ sinh', 'Điện nước', 'Khác'].includes(result.category)
          ? result.category
          : 'Sinh hoạt',
        merchantName: result.merchantName || '',
        items: Array.isArray(result.items) ? result.items : [],
        notes: result.notes || '',
      },
    });
  } catch (error) {
    console.error('Lỗi khi gọi Gemini API quét hóa đơn:', error);
    const msg = error.message || '';
    const isMissingKey = msg.includes('GEMINI_API_KEY');
    const isHighDemand = msg.includes('503') || msg.includes('high demand') || msg.includes('UNAVAILABLE');

    let userFriendlyError = `Lỗi khi nhận diện hóa đơn: ${msg || 'Không thể trích xuất hóa đơn'}`;
    if (isMissingKey) {
      userFriendlyError = 'Chưa cấu hình GEMINI_API_KEY trong hệ thống. Vui lòng thiết lập khóa tại mục Settings > Secrets.';
    } else if (isHighDemand) {
      userFriendlyError = 'Mô hình AI đang có lượng truy cập đột biến tạm thời (503 High Demand). Bạn có thể bấm Thử lại hoặc chuyển sang Tự nhập tay.';
    }

    return res.status(500).json({
      success: false,
      error: userFriendlyError,
    });
  }
});

// Serve static assets from dist or project root
const distPath = path.join(__dirname, 'dist');
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
}
app.use(express.static(__dirname));

// SPA fallback for all routes
app.get('*', (req, res) => {
  const indexInDist = path.join(distPath, 'index.html');
  if (fs.existsSync(indexInDist)) {
    return res.sendFile(indexInDist);
  }
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}`);
});
