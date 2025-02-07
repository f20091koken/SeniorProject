from ..ailogic import base_ai
from ..gameobject import board, box, piece

class GamePlayer:
    def __init__(self, ai=None):
        
        #AIをセット
        self.ai = ai

        #AIの指定がない場合はマニュアル操作を用意
        if ai is None:  self.ai = ManualAi()

    def choice(self, board, box):
        return self.ai.choice(board,box)

    def put(self, board, piece):
        return self.ai.put(board,piece)

class ManualAi(base_ai.BaseAi):
    """
    ManualAiクラス
    手入力をサポートする
    """
    def choice(self, in_board, in_box):
        """
        in_boxリストにある最初の１つを返す
        """
        in_board = board.HiTechBoard(in_board)    #List, Dict形式のデータをBoardクラスに変換
        in_box = box.Box(in_box)            #List, Dict形式のデータをBoxクラスに変換

        boxpiecenum = len(in_box.piecelist)
        box_index = None

        #手入力処理
        while(box_index is None):
            
            # in_str = [int(i) for i in input('choice >>').split()]
            attribute = [int(i) for i in input('choice >>').split()]

            # 属性に一致するインデックスを探す
            for idx, piece in enumerate(in_box.piecelist):
                print(f"Checking piece {idx}: {piece.param.tolist()} against attribute {attribute}")
                if piece.param.tolist() == attribute:
                    in_str = [idx]
                    break
            else:
                print("No such piece")
                continue

            #範囲外の数字を入力
            if not( 0 <= in_str[0] and in_str[0] < boxpiecenum ):continue
            
            box_index = in_str[0]

        #callの判定
        res_call = "Quarto" if in_board.isQuarto() else "Non"

        return {\
            'piece':in_box.piecelist[box_index].toDict(),\
            'call':res_call,\
        }
    
    def put(self, in_board, in_piece):
        """
        in_boardからまだピースが置かれていない座標を返す
        """
        in_board = board.HiTechBoard(in_board)          #List, Dict形式のデータをBoardクラスに変換
        in_piece = piece.Piece.getInstance(in_piece)    #Dict形式のデータをBoardクラスに変換

        res_left = None
        res_top = None

        #手入力処理
        while(res_left is None):
            """"
            image_recognition_result=image_recognition_result()
            print(image_recognition_result)
            print('コマの配置場所があっていたら「cキー」、間違っていたら「rキー」を入力してください')
            while True:
                key = input()  # キー入力を待機
                if key == 'c':  # cが入力されたら続行
                    break
                elif key == 'r':  # rが入力されたら修正処理
                    image_recognition_result = input('修正する駒の配置を入力してください')
            """


            in_str = [int(i) for i in input('put(left top) >>').split()]
            """"
            in_str = [int(i) for i in image_recognition_result.split()]
            """

            #範囲外の数字を入力
            if not (0 <= in_str[0] and in_str[0] <= 3 and 0 <= in_str[1] and in_str[1] <= 3):continue

            #入力箇所に駒がある場合
            if(in_board.getBoard(in_str[0],in_str[1]) is not None): continue

            res_left = in_str[0]
            res_top = in_str[1]

        #コマを置いた上でクアルトするか判定
        in_board.setBoard(res_left,res_top,in_piece)
        res_call = "Quarto" if in_board.isQuarto() else "Non"

        return {\
            'call':res_call,\
            'left':res_left,\
            'top':res_top,\
        }    
