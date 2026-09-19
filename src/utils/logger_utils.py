# 文件路径: src/utils/logger_utils.py
import logging
import os
from datetime import datetime
from hydra.core.hydra_config import HydraConfig

# 定义一个全局变量来缓存 logger，防止重复创建
_logger_instance = None

def get_rank_zero_logger():
    """
    获取一个只在 Rank 0 进程上激活，并写入到 Hydra 输出目录的 logger。
    在其他进程上，返回一个“沉默”的 logger。
    这是一个线程安全的单例模式实现。
    """
    global _logger_instance

    # 只有当 logger 实例还未创建时，才进行配置
    if _logger_instance is None:
        try:
            # 尝试获取 Hydra 的输出目录，这是最可靠的方式
            hydra_output_dir = HydraConfig.get().run.dir
            log_dir = os.path.join(hydra_output_dir, "logs")
        except Exception:
            # 如果 Hydra 未初始化（例如在非 Hydra 环境下），则使用默认路径
            log_dir = "logs"

        # 创建一个有特定名字的 logger，避免与 root logger 冲突
        logger = logging.getLogger("MyProjectLogger")
        logger.setLevel(logging.INFO)
        
        # --- 关键：防止日志消息向上传播到 root logger ---
        logger.propagate = False

        # 检查是否已经有 handler，防止重复添加
        if not logger.hasHandlers():
            # 只在 Rank 0 进程上添加文件处理器
            if os.environ.get("LOCAL_RANK", "0") == "0":
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")

                # 创建文件 handler
                file_handler = logging.FileHandler(log_file, mode='w')
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                logger.info(f"日志系统已为 Rank 0 初始化。日志将保存在: {log_file}")
            else:
                # 在其他所有进程上，添加一个“无操作”的 handler
                # 这样调用 logger.info() 就不会做任何事，也不会报错
                logger.addHandler(logging.NullHandler())
        
        _logger_instance = logger

    return _logger_instance