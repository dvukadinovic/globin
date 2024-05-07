.. _keywords:


'params.input' file keywords
=============================

'keyword.input' file keywords
=============================

'LS_Lande' : bool, default True
	Flag if the Lande factors for each level in Kurucz line list should be computed in the LS coupling sheme. For 'LS_Lande'=False, we read explicitly Lande factors for each level. If the level has the Lande factor -99, it is to be compute internaly in the LS coupling sheme. Others can originate from any other coupling sheme (JJ, JK,...) depending on the couopling of the level. 
	
