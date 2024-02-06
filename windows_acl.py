import win32security
import ntsecuritycon as con
import win32con

def revoke_read_permission(filename):
    # ファイル/フォルダのセキュリティ情報を取得
    sd = win32security.GetFileSecurity(filename, win32security.DACL_SECURITY_INFORMATION)
    
    # 新しいDACLを作成
    dacl = win32security.ACL()
    
    # ファイル/フォルダの現在のDACLを取得
    _, dacl, _ = sd.GetSecurityDescriptorDacl()
    
    # DACLを走査して、読み取り権限を持つエントリを削除
    for i in range(dacl.GetAceCount()):
        ace = dacl.GetAce(i)
        if ace[1][0] & (win32con.FILE_GENERIC_READ | win32con.FILE_EXECUTE):
            # 読み取りおよび実行権限を持つエントリを削除
            dacl.DeleteAce(i)
    
    # 更新されたDACLをセキュリティ記述子に設定
    sd.SetSecurityDescriptorDacl(1, dacl, 0)
    
    # ファイル/フォルダのセキュリティ情報を更新
    win32security.SetFileSecurity(filename, win32security.DACL_SECURITY_INFORMATION, sd)

filename = "path_to_your_file_or_folder"
revoke_read_permission(filename)
