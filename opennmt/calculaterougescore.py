##Open source courtesy : https://github.com/pltrdy/rouge/

from rouge import FilesRouge

def CalculateRouge(ref, hyp):
  files_rouge = FilesRouge(ref, hyp)
  scores = files_rouge.get_scores(avg= True)
  print(scores)




if __name__ == "__main__":
    reference_list = ['base-gold.txt', 'short-gold.txt', 'embed-gold.txt', 'long-gold.txt', 'pos-gold.txt']
    hypothesis_list = ['base-pred.txt', 'short-pred.txt', 'embed-pred.txt', 'long-pred.txt', 'pos-pred.txt']

    index = 0
    for gold in reference_list:
      pred = hypothesis_list[index]
      print(gold)
      print(pred)
      CalculateRouge(gold, pred)
      index +=1
      #out = open('bleu_out.txt', 'w')
      #  out.write(str(bleu))
      #  out.close()



