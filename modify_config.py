#用于批量修改数据库中保存的默认配置
import yaml
import pymysql

def mod_config():
    config_string = """
    history:
        start: null
        end: null
        override: True
    incremental:
        override: True
    """

    # 解析配置字符串
    config = yaml.safe_load(config_string)

    # 添加新的配置项
    config['incremental']['date_offset'] = 0
    config['data_restrict'] = True

    # 转换为 YAML 格式的字符串
    config_string = yaml.dump(config)

    # 输出字符串
    print(config_string)
    return config_string