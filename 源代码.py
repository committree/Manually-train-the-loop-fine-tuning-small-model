import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

POS_TEMPLATES = [
    "东西收到了，质量很好，物流也快",
    "性价比超高，用了一周没有任何问题",
    "包装很仔细，客服态度也好，好评",
    "比想象中好用，会回购的",
    "做工精致，颜色和图片一致，很满意",
    "发货速度快，第二天就到了，赞",
    "手感不错，功能齐全，值得购买",
    "朋友推荐买的，确实没让我失望",
    "价格实惠，质量对得起这个价",
    "用起来很顺手，家里人都说好",
    "细节做得很好，看得出用心了",
    "第二次买了，一如既往地好",
    "完全符合描述，非常满意的一次购物",
    "外观漂亮，性能也强，强烈推荐",
    "质量比预期还好，商家很实在",
]
NEG_TEMPLATES = [
    "包装破损，客服态度差，差评",
    "用了两天就坏了，质量太差",
    "和描述完全不符，退货了",
    "物流慢得要命，东西还有瑕疵",
    "做工粗糙，边角都是毛刺",
    "收到就是二手的，太失望了",
    "价格贵还不好用，不推荐",
    "颜色和图片差太多，货不对板",
    "用了就出问题，售后还推脱",
    "一点都不值这个价，后悔买",
    "质量堪忧，用一次就想扔",
    "客服爱答不理，体验极差",
    "尺寸完全不对，白等了",
    "有异味，材质一看就很差",
    "第三次买了这次翻车，不会再来",
]
PREFIXES = ["", "说实话，", "整体来说，", "刚收到，", "嗯，"]
SUFFIXES = ["", "。", "！", "，就这样吧", "，给个参考"]

def build_dataset(n_total=200):
    text, labels = [], []
    half = n_total // 2
    for _ in range(half):
        s = random.choice(PREFIXES) + random.choice(POS_TEMPLATES) + random.choice(SUFFIXES)
        text.append(s);labels.append(1)
    for _ in range(n_total - half):
        s = random.choice(PREFIXES) + random.choice(NEG_TEMPLATES) + random.choice(SUFFIXES)
        text.append(s)
        labels.append(0)

    data = list(zip(text, labels))
    random.shuffle(data)
    text, labels = zip(*data)
    return list(text), list(labels)

all_texts, all_labels = build_dataset(200)
print(f"数据总量: {len(all_texts)}  正样本: {sum(all_labels)}  负样本: {len(all_labels)-sum(all_labels)}")

MODEL_NAME = "hf1/rbt3"
MAX_LEN = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,max_length=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return{
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

full_ds = SentimentDataset(all_texts, all_labels, tokenizer)
n_val = int(0.2 * len(full_ds))
n_train = len(full_ds) - n_val
train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

BATCH_SIZE = 16
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
print(f"训练集: {len(train_ds)}  验证集: {len(val_ds)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2)
model.to(device)

EPOCHS = 3
LR = 2e-5
optimizer = AdamW(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        predictions = torch.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return correct / total, all_predictions, all_labels

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
        loss = outputs.loss         
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    acc, _, _ = evaluate(model, val_loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] loss={total_loss/len(train_loader):.4f} | val_acc={acc:.4f}")

model.save_pretrained("./rbt3_sentiment")
tokenizer.save_pretrained("./rbt3_sentiment")
print("模型已保存到 ./rbt3_sentiment")

def predict(text):
    model.eval()
    enc = tokenizer(text, truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
    pred = logits.argmax(dim=-1).item()
    return "正面" if pred == 1 else "负面"

for t in ["这质量真不错，下次还来", "垃圾东西，用一次就坏", "一般般吧，凑合能用"]:
    print(f"{t}  ->  {predict(t)}")


