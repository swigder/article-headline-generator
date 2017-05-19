## Please place (Unprocesed files) in the same folder: 
	# 1) translations-base.txt
	# 2) translations-embed.txt
	# 3) translations-long.txt
	# 4) translations-pos.txt
	# 5) translations-short.txt
	# 6) calculatebleu.py
	# 7) calculaterougescore.py


################ Step 1: Process Translation files ##################
python3 process_translation_output.py --m 'base' translations-base.txt
python3 process_translation_output.py --m 'embed' translations-embed.txt
python3 process_translation_output.py --m 'long' translations-long.txt
python3 process_translation_output.py --m 'pos' translations-pos.txt
python3 process_translation_output.py --m 'short' translations-short.txt
###########################################################

############## Step 2: Bleu score calculation #################
echo "************************** BLEW SCORE START *********************"
echo "bleu score for base(below):"
python3 calculatebleu.py base-gold.txt base-pred.txt
echo "bleu score for embed(below):"
python3 calculatebleu.py embed-gold.txt embed-pred.txt
echo "bleu score for long(below):"
python3 calculatebleu.py long-gold.txt long-pred.txt
echo "bleu score for pos(below):"
python3 calculatebleu.py pos-gold.txt pos-pred.txt
echo "bleu score for short(below):"
python3 calculatebleu.py short-gold.txt short-pred.txt
echo "************************* BLEW SCORE END *************************"
##############################################################

############ Step 3: Rogue score calculation ###############
echo "********************** ROGUE SCORES START ************************"
##Please note: There is a mild incosistency! Unlike blew score calculations, the file name list has been already fed in to the calculateroguescore.py file
python3 calculaterougescore.py
echo "********************** ROGUE SCORES END ************************"
############################################################


