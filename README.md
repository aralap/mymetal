# mymetal
Metal Binding Prediction
##Metal Binding Predictor, using neural network

#A simple way to test:

from mymetal import mbp

mbp.predict('fasta_file')

This will return a list, first 3 lines are a summary of the hits and the rest are predictions for each protein supplied.

-----------

In case you wish to output a CSV file:

from mymetal import mbp,iof

a = mbp.predict('fasta_file')
iof.save_out_csv(a,'out_filename.csv')
