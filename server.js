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

// =========================================================================
// API: ĐỐI SOÁT BILL CHUYỂN KHOẢN NGÂN HÀNG BẰNG GEMINI MULTIMODAL AI
// =========================================================================
app.post('/api/reconcile-payment', async (req, res) => {
  try {
    const { imageBase64, mimeType, roomId, expectedCode, expectedAmount } = req.body;
    if (!imageBase64) {
      return res.status(400).json({
        success: false,
        error: 'Vui lòng cung cấp hình ảnh bill chuyển khoản cần đối soát.',
      });
    }

    const ai = getGeminiClient();

    // Chuẩn hóa base64 thuần
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

    const promptText = `
Bạn là một AI chuyên gia đối soát tài chính KTX qua hình ảnh hóa đơn (bill chuyển khoản ngân hàng Việt Nam như Vietcombank, Techcombank, MB Bank, TPBank, VPBank, VietinBank, BIDV, MoMo, ZaloPay, Cake...).
Hãy quét toàn bộ hình ảnh bill đính kèm và trích xuất các thông tin chuyển khoản:

Quy tắc trích xuất:
1. Chỉ chấp nhận nếu trạng thái giao dịch là Thành công (Thanh toán thành công, Giao dịch thành công, Chuyển tiền thành công, Chuyển khoản thành công, Thành công).
2. Tìm số tiền chuyển khoản chính xác (bằng số nguyên VNĐ, ví dụ: 26767, 50000, 120000...).
3. Tìm mã đối soát khoản chi hoặc nội dung chuyển khoản trong phần lời nhắn/nội dung giao dịch (Ví dụ: chuỗi dạng MAKC992, KTX402 PAY HUY, KTX402, hoặc mã tương đương).
4. Lấy tên người nhận, người chuyển và ngân hàng nếu có.
5. Nếu ảnh không phải bill ngân hàng hoặc giao dịch thất bại / đang xử lý / hủy, đặt transaction_status thành "IGNORE".
`;

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
            text: promptText,
          },
        ],
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            transaction_status: {
              type: Type.STRING,
              description: 'SUCCESS nếu giao dịch thành công hợp lệ, ngược lại IGNORE',
            },
            amount: {
              type: Type.NUMBER,
              description: 'Số tiền chuyển khoản bằng số nguyên VNĐ',
            },
            sender_name: {
              type: Type.STRING,
              description: 'Tên người gửi / người chuyển nếu có',
            },
            receiver_name: {
              type: Type.STRING,
              description: 'Tên người nhận tiền nếu có',
            },
            bank_name: {
              type: Type.STRING,
              description: 'Tên ngân hàng chuyển tiền hoặc thụ hưởng',
            },
            payment_code: {
              type: Type.STRING,
              description: 'Nội dung chuyển khoản hoặc mã đối soát (ví dụ: KTX402 PAY..., MAKC...)',
            },
            transaction_id: {
              type: Type.STRING,
              description: 'Mã tham chiếu / mã giao dịch ngân hàng nếu có',
            },
          },
          required: ['transaction_status', 'amount'],
        },
      },
    });

    console.log(`Đã đối soát bill qua model: ${modelUsed}`);
    const rawText = response.text ? response.text.trim() : '{}';
    let result = {};
    try {
      result = JSON.parse(rawText);
    } catch (e) {
      console.warn('Lỗi parse JSON từ Gemini response:', e.message, rawText);
    }

    const isSuccess = (result.transaction_status || '').toUpperCase() === 'SUCCESS';
    const parsedAmount = Math.round(Number(result.amount) || 0);
    const paymentCode = (result.payment_code || '').trim();

    return res.json({
      success: true,
      data: {
        transaction_status: isSuccess ? 'SUCCESS' : 'IGNORE',
        amount: parsedAmount,
        sender_name: result.sender_name || '',
        receiver_name: result.receiver_name || '',
        bank_name: result.bank_name || '',
        payment_code: paymentCode,
        transaction_id: result.transaction_id || '',
        model_used: modelUsed,
      },
    });
  } catch (error) {
    console.error('Lỗi khi gọi Gemini API đối soát bill:', error);
    const msg = error.message || '';
    const isMissingKey = msg.includes('GEMINI_API_KEY');
    const isHighDemand = msg.includes('503') || msg.includes('high demand') || msg.includes('UNAVAILABLE');

    let userFriendlyError = `Lỗi khi đối soát bill chuyển khoản: ${msg || 'Không thể đọc ảnh bill'}`;
    if (isMissingKey) {
      userFriendlyError = 'Chưa cấu hình GEMINI_API_KEY. Vui lòng thiết lập khóa tại Settings > Secrets.';
    } else if (isHighDemand) {
      userFriendlyError = 'Mô hình AI đang có lượng truy cập cao tạm thời (503). Vui lòng thử lại sau vài giây.';
    }

    return res.status(500).json({
      success: false,
      error: userFriendlyError,
    });
  }
});

// =========================================================================
// REAL-TIME ROOM SYNC ENGINE (Server-authoritative + Multi-device SSE stream)
// =========================================================================
const dataDir = path.join(__dirname, 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}
const dbFilePath = path.join(dataDir, 'rooms_db.json');

const roomsCache = new Map();
const activitiesCache = new Map(); // roomId -> Array
const sseClients = new Map(); // roomId -> Set of res objects

// Helper: load persisted rooms
function loadPersistedRooms() {
  try {
    if (fs.existsSync(dbFilePath)) {
      const raw = fs.readFileSync(dbFilePath, 'utf8');
      const json = JSON.parse(raw);
      if (json.rooms && typeof json.rooms === 'object') {
        for (const [id, val] of Object.entries(json.rooms)) {
          roomsCache.set(id.toLowerCase(), val);
        }
      }
      if (json.activities && typeof json.activities === 'object') {
        for (const [id, val] of Object.entries(json.activities)) {
          activitiesCache.set(id.toLowerCase(), Array.isArray(val) ? val : []);
        }
      }
    }
  } catch (err) {
    console.warn('Lỗi đọc rooms_db.json:', err.message);
  }
}
loadPersistedRooms();

function savePersistedRooms() {
  try {
    const obj = {
      rooms: Object.fromEntries(roomsCache.entries()),
      activities: Object.fromEntries(activitiesCache.entries()),
      lastSaved: Date.now()
    };
    fs.writeFileSync(dbFilePath, JSON.stringify(obj, null, 2), 'utf8');
  } catch (err) {
    console.warn('Lỗi ghi rooms_db.json:', err.message);
  }
}

function broadcastSse(roomId, eventName, payload) {
  const normId = (roomId || '').toLowerCase().trim();
  const clients = sseClients.get(normId);
  if (!clients || clients.size === 0) return;

  const dataStr = `event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`;
  for (const client of clients) {
    try {
      client.write(dataStr);
    } catch (e) {
      clients.delete(client);
    }
  }
}

// 1. SSE Stream: Nhận thay đổi tức thì cho điện thoại & laptop
app.get('/api/rooms/:roomId/events', (req, res) => {
  const normId = (req.params.roomId || 'phong-402').toLowerCase().trim();

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.flushHeaders();

  if (!sseClients.has(normId)) {
    sseClients.set(normId, new Set());
  }
  const clientSet = sseClients.get(normId);
  clientSet.add(res);

  // Gửi ngay trạng thái hiện tại của phòng nếu có
  const currentRoom = roomsCache.get(normId);
  if (currentRoom) {
    res.write(`event: room_sync\ndata: ${JSON.stringify(currentRoom)}\n\n`);
  }

  // Heartbeat ping mỗi 20s để giữ kết nối trên mạng di động 4G/WiFi
  const pingTimer = setInterval(() => {
    try {
      res.write(': ping\n\n');
    } catch (e) {
      clearInterval(pingTimer);
      clientSet.delete(res);
    }
  }, 20000);

  req.on('close', () => {
    clearInterval(pingTimer);
    clientSet.delete(res);
  });
});

// 2. Lấy dữ liệu phòng
app.get('/api/rooms/:roomId', async (req, res) => {
  const normId = (req.params.roomId || 'phong-402').toLowerCase().trim();
  let room = roomsCache.get(normId);

  // Nếu trong cache chưa có, thử tìm theo alias hoặc phòng phong-402
  if (!room && normId !== 'phong-402') {
    const defaultRoom = roomsCache.get('phong-402');
    if (defaultRoom && (defaultRoom.roomCode || '').toLowerCase() === normId) {
      room = defaultRoom;
      roomsCache.set(normId, room);
    }
  }

  // Thử kéo từ Firestore nếu server chưa có trong bộ nhớ
  if (!room) {
    try {
      const configPath = path.join(__dirname, 'firebase-applet-config.json');
      if (fs.existsSync(configPath)) {
        const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
        const dbId = config.firestoreDatabaseId || '(default)';
        const fsUrl = `https://firestore.googleapis.com/v1/projects/${config.projectId}/databases/${dbId}/documents/ktx_rooms/${normId}?key=${config.apiKey}`;
        const fsRes = await fetch(fsUrl);
        if (fsRes.ok) {
          const docJson = await fsRes.json();
          if (docJson.fields) {
            // Chuyển đổi Firestore fields sang object thông thường
            const parsed = {
              roomId: docJson.fields.roomId?.stringValue || normId,
              roomName: docJson.fields.roomName?.stringValue || `Phòng ${normId.toUpperCase()}`,
              roomCode: docJson.fields.roomCode?.stringValue || normId.toUpperCase(),
              updatedAtMs: Number(docJson.fields.updatedAtMs?.integerValue || docJson.fields.updatedAtMs?.doubleValue) || Date.now(),
              updatedBy: docJson.fields.updatedBy?.stringValue || 'Bạn cùng phòng',
              senderDeviceId: docJson.fields.senderDeviceId?.stringValue || 'cloud'
            };
            if (docJson.fields.members?.arrayValue?.values) {
              parsed.members = docJson.fields.members.arrayValue.values.map(v => ({
                id: v.mapValue?.fields?.id?.stringValue || 'm_' + Math.random(),
                name: v.mapValue?.fields?.name?.stringValue || '',
                bank: v.mapValue?.fields?.bank?.stringValue || '',
                account: v.mapValue?.fields?.account?.stringValue || ''
              }));
            }
            if (docJson.fields.expenses?.arrayValue?.values) {
              parsed.expenses = docJson.fields.expenses.arrayValue.values.map(v => {
                const f = v.mapValue?.fields || {};
                return {
                  id: f.id?.stringValue || 'e_' + Math.random(),
                  name: f.name?.stringValue || '',
                  amount: Number(f.amount?.integerValue || f.amount?.doubleValue || 0),
                  category: f.category?.stringValue || 'Sinh hoạt',
                  payerId: f.payerId?.stringValue || '',
                  beneficiaries: (f.beneficiaries?.arrayValue?.values || []).map(b => b.stringValue),
                  date: f.date?.stringValue || new Date().toISOString()
                };
              });
            }
            room = parsed;
            roomsCache.set(normId, room);
            savePersistedRooms();
          }
        }
      }
    } catch (e) {
      console.warn('Không thể đọc Firestore fallback:', e.message);
    }
  }

  if (room) {
    return res.json({ success: true, data: room });
  }

  return res.status(404).json({ success: false, message: 'Phòng chưa có dữ liệu trên máy chủ' });
});

// 3. Cập nhật và đồng bộ dữ liệu phòng
app.post('/api/rooms/:roomId', (req, res) => {
  const normId = (req.params.roomId || 'phong-402').toLowerCase().trim();
  const payload = req.body;

  if (!payload || typeof payload !== 'object') {
    return res.status(400).json({ success: false, error: 'Dữ liệu không hợp lệ' });
  }

  const existing = roomsCache.get(normId);
  const nowMs = Date.now();
  const incomingTime = Number(payload.updatedAtMs) || nowMs;

  // Cập nhật timestamp và id
  payload.roomId = normId;
  payload.updatedAtMs = incomingTime;

  roomsCache.set(normId, payload);

  // Nếu roomCode khác normId, lưu thêm vào alias để thiết bị gõ mã phòng nào cũng tìm thấy
  if (payload.roomCode) {
    const aliasCode = payload.roomCode.toLowerCase().trim().replace(/[^a-zA-Z0-9_-]/g, '-');
    if (aliasCode && aliasCode !== normId) {
      roomsCache.set(aliasCode, payload);
    }
  }

  savePersistedRooms();

  // Phát sóng lập tức cho tất cả client đang mở phòng này
  broadcastSse(normId, 'room_sync', payload);
  if (payload.roomCode) {
    const aliasCode = payload.roomCode.toLowerCase().trim().replace(/[^a-zA-Z0-9_-]/g, '-');
    if (aliasCode && aliasCode !== normId) {
      broadcastSse(aliasCode, 'room_sync', payload);
    }
  }

  return res.json({ success: true, updatedAtMs: incomingTime });
});

// 4. Bảng tin hoạt động nhanh của phòng
app.get('/api/rooms/:roomId/activities', (req, res) => {
  const normId = (req.params.roomId || 'phong-402').toLowerCase().trim();
  const acts = activitiesCache.get(normId) || [];
  return res.json({ success: true, data: acts.slice(-30).reverse() });
});

app.post('/api/rooms/:roomId/activities', (req, res) => {
  const normId = (req.params.roomId || 'phong-402').toLowerCase().trim();
  const { author, type, description } = req.body || {};

  if (!description) {
    return res.status(400).json({ success: false, error: 'Thiếu nội dung' });
  }

  let acts = activitiesCache.get(normId);
  if (!acts) {
    acts = [];
    activitiesCache.set(normId, acts);
  }

  const newAct = {
    id: 'act_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6),
    author: author || 'Bạn cùng phòng',
    type: type || 'notice',
    description: description,
    createdAt: new Date().toISOString()
  };

  acts.push(newAct);
  if (acts.length > 50) acts.shift();
  savePersistedRooms();

  broadcastSse(normId, 'activity_new', newAct);
  return res.json({ success: true, data: newAct });
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
