import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import os

# Giả định lớp Config được import từ configs/config.py
# from configs.config import Config 

class BaseModel(nn.Module, ABC):
    """
    Lớp mô hình cơ sở (Base Model Class) cho tất cả các mô hình Dịch máy
    trong dự án. Mọi mô hình con phải kế thừa từ lớp này và triển khai
    các phương thức trừu tượng.
    """
    
    def __init__(self, config: Dict[str, Any], src_vocab_size: int, tgt_vocab_size: int):
        """
        Khởi tạo BaseModel.
        
        Args:
            config (Dict): Cấu hình chung cho mô hình (từ file YAML/Config).
            src_vocab_size (int): Kích thước từ vựng ngôn ngữ nguồn (Source).
            tgt_vocab_size (int): Kích thước từ vựng ngôn ngữ đích (Target).
        """
        super().__init__()
        self.config = config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Các thuộc tính chung (ví dụ: kích thước embedding)
        self.embed_dim = config.get("model.embed_dim", 512)
        self.dropout_rate = config.get("model.dropout", 0.1)
        
        # Khởi tạo lớp embedding (dùng chung cho Encoder và Decoder, nếu cần)
        # Lưu ý: Các mô hình con có thể tự định nghĩa embedding riêng nếu cần thiết
        self.src_embedding = nn.Embedding(src_vocab_size, self.embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, self.embed_dim)

        # Lớp Linear cuối cùng (Projection Layer) để chuyển đổi output thành kích thước vocab đích
        self.output_projection = nn.Linear(self.embed_dim, self.tgt_vocab_size)

    # --- Phương thức Trừu tượng (Bắt buộc các mô hình con phải triển khai) ---

    @abstractmethod
    def forward(self, 
                src_seq: torch.Tensor, 
                tgt_seq: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        Pass Forward chính của mô hình trong quá trình Huấn luyện.
        Phải được triển khai bởi các mô hình con (LSTM/Transformer).
        
        Args:
            src_seq (Tensor): Chuỗi đầu vào nguồn (batch_size, src_len).
            tgt_seq (Tensor): Chuỗi đầu vào đích (batch_size, tgt_len).
            src_mask (Tensor, optional): Mask cho chuỗi nguồn (padding mask).
            tgt_mask (Tensor, optional): Mask cho chuỗi đích (padding và look-ahead mask).
            
        Returns:
            Tensor: Logits (unnormalized scores) của từ vựng đích (batch_size, tgt_len, tgt_vocab_size).
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, 
               src_seq: torch.Tensor, 
               src_mask: Optional[torch.Tensor] = None
              ) -> Any:
        """
        Thực hiện mã hóa chuỗi nguồn (Encoder).
        Phải được triển khai bởi các mô hình con.
        
        Returns:
            Any: Output của Encoder (ví dụ: hidden states, context vector,...).
        """
        raise NotImplementedError

    @abstractmethod
    def decode_step(self, 
                    tgt_token: torch.Tensor, 
                    encoder_output: Any,
                    past_key_values: Optional[Any] = None
                   ) -> Tuple[torch.Tensor, Any]:
        """
        Thực hiện một bước dịch (decoding step) duy nhất (sử dụng cho inference/translation).
        
        Returns:
            Tuple[Tensor, Any]: Logits của từ tiếp theo và các trạng thái/cache mới.
        """
        raise NotImplementedError

    def save(self, path: str, epoch: Optional[int] = None, step: Optional[int] = None):
        """
        Lưu trạng thái (state_dict) của mô hình PyTorch vào file.
        
        Args:
            path (str): Đường dẫn cơ sở để lưu file.
            epoch (int, optional): Epoch hiện tại (dùng để đặt tên file).
            step (int, optional): Bước hiện tại (dùng để đặt tên file).
        """
        # Tạo tên file
        if epoch is not None and step is not None:
            filename = f"model_epoch{epoch:03d}_step{step}.pt"
        elif epoch is not None:
            filename = f"model_epoch{epoch:03d}.pt"
        else:
            filename = "model_latest.pt"
            
        full_path = os.path.join(path, filename)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(path, exist_ok=True)
        
        # Lưu state_dict của mô hình
        torch.save(self.state_dict(), full_path)
        print(f"✅ Đã lưu mô hình tới: {full_path}")

    def load(self, full_path: str, device: str = 'cpu'):
        """
        Tải trọng số (state_dict) của mô hình từ file.
        
        Args:
            full_path (str): Đường dẫn đầy đủ tới file trọng số (.pt).
            device (str): Thiết bị (CPU/GPU) để tải trọng số vào.
        
        Raises:
            FileNotFoundError: Nếu file trọng số không tồn tại.
        """
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"❌ Lỗi: Không tìm thấy file trọng số tại {full_path}")
            
        # Tải state_dict
        state_dict = torch.load(full_path, map_location=device)
        
        # Tải trọng số vào mô hình hiện tại
        self.load_state_dict(state_dict)
        print(f"✅ Đã tải mô hình từ: {full_path} trên thiết bị {device}")

    def predict(self, src_seq: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        """
        Thực hiện quá trình dịch (Inference) sử dụng Greedy Decoding. 
        Phương thức này được đặt tên là `predict` và sử dụng `translate`
        làm logic lõi.
        
        Args:
            src_seq (Tensor): Chuỗi đầu vào nguồn (batch_size, src_len).
            max_len (int): Chiều dài tối đa của chuỗi đích.
        
        Returns:
            Tensor: Chuỗi token đã dịch (batch_size, translated_len).
        """
        # Đây chỉ là một wrapper đơn giản, gọi hàm translate đã được định nghĩa
        return self.translate(src_seq, max_len=max_len)
    
    # --- Phương thức Chung (Có thể dùng chung, không bắt buộc ghi đè) ---

    def translate(self, src_seq: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        """
        Phương thức dịch (Inference) sử dụng Greedy Decoding. 
        Mô hình con có thể ghi đè phương thức này bằng Beam Search.
        """
        self.eval() # Chuyển sang chế độ đánh giá
        with torch.no_grad():
            
            # 1. Mã hóa đầu vào nguồn
            encoder_output = self.encode(src_seq)
            
            batch_size = src_seq.size(0)
            
            # Giả định <SOS> token có ID là 1 (Start Of Sentence)
            # Khởi tạo chuỗi đích bằng <SOS>
            start_token_id = self.config.get("data.sos_id", 1) 
            end_token_id = self.config.get("data.eos_id", 2)
            
            # output_tokens: (batch_size, 1)
            output_tokens = torch.full((batch_size, 1), start_token_id, 
                                       dtype=torch.long, device=src_seq.device)
            
            # Các trạng thái/cache cần thiết cho Decoder
            decoder_state = None 
            
            # 2. Vòng lặp dịch
            for _ in range(max_len - 1):
                # Lấy token cuối cùng đã được dịch
                last_token = output_tokens[:, -1] # (batch_size)
                
                # Thực hiện một bước giải mã
                logits, decoder_state = self.decode_step(last_token, encoder_output, decoder_state)
                
                # Lấy chỉ số (index) của từ có xác suất cao nhất
                next_word_token = torch.argmax(logits, dim=-1) # (batch_size)
                
                # Nối token mới vào chuỗi kết quả
                output_tokens = torch.cat([output_tokens, next_word_token.unsqueeze(-1)], dim=-1)
                
                # Điều kiện dừng: Nếu tất cả các chuỗi đã kết thúc bằng <EOS>
                if (next_word_token == end_token_id).all():
                    break

        return output_tokens.detach()
    

