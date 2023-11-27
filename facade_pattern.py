"""
こんな処理構成にFacade設計は適任でしょうか
１．データ更新用明細がExcelシートで送信される
２．データのバリデーションチェック
３．Excel データ列から新たなColumnsを生成する
４．他データとマージしてデータ補填する
５．MasterデータのDataFrameをExcel明細で更新する
６．Masterデータから更新後データをCSVファイルで作成する
"""

class DataUpdateSubSystem:
    def __init__(self):
        # 細かい処理はprivateメソッドにcapsule化

    def validate(self, excel): 
        # ステップ2 

    def generate_columns(self, excel):
        # ステップ3

    def enrich(self, data):
        # ステップ4

    def update_master(self, data):
        # ステップ5 

    def export_csv(self, master):
        # ステップ6

# ファサード
class DataUpdateFacade:
    def __init__(self):
        self._subsystem = DataUpdateSubSystem()

    def update(self, excel):
        data = self._subsystem.validate(excel)
        data = self._subsystem.generate_columns(data)
        data = self._subsystem.enrich(data)
        self._subsystem.update_master(data)
        self._subsystem.export_csv(data)
