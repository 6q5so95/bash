Sub ControlInternetExplorer()
    Dim IE As Object ' InternetExplorerオブジェクト
    Dim URL As String ' 開くURL
    
    ' InternetExplorerオブジェクトを作成
    Set IE = CreateObject("InternetExplorer.Application")
    
    ' IEウィンドウを非表示にする場合は次の行をコメントアウト
    ' IE.Visible = True
    
    ' 開くURLを指定
    URL = "https://www.example.com"
    
    ' ウィンドウを最大化
    IE.FullScreen = True
    
    ' 指定したURLを開く
    IE.Navigate URL
    
    ' ページの読み込みが完了するまで待機
    Do While IE.Busy Or IE.readyState <> 4
        DoEvents
    Loop
    
    ' ページのタイトルを表示
    MsgBox IE.Document.Title
    
    ' InternetExplorerオブジェクトを解放
    Set IE = Nothing
End Sub
