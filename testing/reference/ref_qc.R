# Reference harness for the test suite.
#
# The QC algorithm below is copied verbatim from PWSQC_2.R and PWSQC_3.R of the
# reference implementation (https://github.com/LottedeVos/PWSQC, CC-BY-SA 4.0 by
# Lotte de Vos). Only the file handling around it was replaced, so that the
# parameters and the data can be passed in as CSV instead of RData.
# usage: Rscript ref_qc.R <dir> <range> <nstat> <nint> <HIthresA> <HIthresB>
#                         <compareint> <rainyint> <matchint> <corthres>
#                         <defaultbiascorrection> <biasthres>
args <- commandArgs(trailingOnly=TRUE)
dir <- args[1]
nstat <- as.numeric(args[3]); nint <- as.numeric(args[4])
HIthresA <- as.numeric(args[5]); HIthresB <- as.numeric(args[6])
compareint <- as.numeric(args[7]); rainyint <- as.numeric(args[8])
matchint <- as.numeric(args[9]); corthres <- as.numeric(args[10])
defaultbiascorrection <- as.numeric(args[11]); biasthres <- as.numeric(args[12])

Nraw <- read.csv(file.path(dir, 'Ndataset.csv'), check.names=FALSE)
Ndataset <- as.matrix(Nraw[, -1, drop=FALSE])
Meta <- data.frame(id=as.numeric(colnames(Nraw)[-1]))

nb <- read.csv(file.path(dir, 'neighbourlist.csv'), colClasses='character')
neighbourlist <- vector('list', nrow(Meta))
for (i in 1:nrow(Meta)) {
    row <- nb[which(as.numeric(nb$station_id) == Meta$id[i]), ]
    s <- row$neighbours[1]
    if (is.na(s) || nchar(s) == 0) { neighbourlist[[i]] <- numeric(0) }
    else { neighbourlist[[i]] <- as.numeric(strsplit(s, ',')[[1]]) }
}

# ------------------------------------------------------------------ PWSQC_2.R
for(i in 1:nrow(Meta)){
	Nint <- Ndataset[,i]
	if((length(which(is.na(Nint)==F)) < 1) | (length(neighbourlist[[i]]) < nstat)){
		  HIflag <- FZflag <- rep(-1, times=length(Nint))
   		  if(exists("HI_flags")==F){ HI_flags <- HIflag
			}else{ HI_flags <- cbind(HI_flags, HIflag) }
 		  if(exists("FZ_flags")==F){ FZ_flags <- FZflag
			}else{ FZ_flags <- cbind(FZ_flags, FZflag) }
	}else{

   NeighbourVal <- Ndataset[,which(Meta$id %in% neighbourlist[[i]])]

   Ref <- rep(NA, times=length(Nint))
   Number_of_measurements <- apply(NeighbourVal, 1, function(x) length(which(is.na(x)==F)))
   Med <- apply(NeighbourVal, 1, median, na.rm=T)

   # # # HI-filter:
   HIflag <- rep(0, times=length(Nint))
   HIflag[which(((Nint > HIthresB) & (Med < HIthresA)) | ((Med >= HIthresA) & (Nint > (HIthresB*Med/HIthresA))))] <- 1
   HIflag[which(Number_of_measurements < nstat)] <- -1

   if(exists("HI_flags")==F){ HI_flags <- HIflag
	}else{ HI_flags <- cbind(HI_flags, HIflag) }

   # # # FZ-filter:
   Ref[which(Med == 0)] <- 0
   Ref[which(Med >  0)] <- 1
   Ref[which(Number_of_measurements < nstat)] <- NA

   Nwd <- Nint
   Nwd[which(Nint > 0)] <- 1
   runs <- rle(Nwd)
   rownr <- cumsum(runs$lengths)
   endrow <- rownr[ which(runs$lengths > nint & runs$values==0) ]
   startrow <- endrow - runs$lengths[ which(runs$lengths > nint  & runs$values==0) ] + 1

   FZflag <- rep(0, times=length(Nint))
   if(length(endrow) > 0){
   for(r in 1:length(endrow)){
   	if(length( which( (Ref[startrow[r] : endrow[r]]) == 1) ) > nint ){

	runs2 <- rle(Ref[startrow[r] : endrow[r]])
   	rownr2 <- cumsum(runs2$lengths)
   	endrow2 <- rownr2[ which(runs2$lengths > nint & runs2$values==1) ]
   	startrow2 <- endrow2 - runs2$lengths[ which(runs2$lengths > nint  & runs2$values==1) ] + 1

	if(length(startrow2) > 0){
	FZstartrow <- startrow[r] + startrow2[1] - 1 + nint

   	FZflag[FZstartrow : endrow[r]] <- 1

	m <- 1
	while((is.na(Nwd[endrow[r] + m])|(Nwd[endrow[r] + m] == 0)) & ((endrow[r]+m) <= length(Nwd)) ){
	 FZflag[endrow[r]+m] <- 1
	 m <- m+1
	}

	}}
   }
   }

   FZflag[which(Number_of_measurements < nstat)] <- -1
   if(exists("FZ_flags")==F){ FZ_flags <- FZflag
	}else{ FZ_flags <- cbind(FZ_flags, FZflag) }

	}
}
FZ_flags <- matrix(FZ_flags, ncol=nrow(Meta)); HI_flags <- matrix(HI_flags, ncol=nrow(Meta))
write.csv(FZ_flags, file.path(dir, 'FZ_flags.csv'), row.names=FALSE)
write.csv(HI_flags, file.path(dir, 'HI_flags.csv'), row.names=FALSE)

# ------------------------------------------------------------------ PWSQC_3.R
Ndataset2 <- Ndataset * defaultbiascorrection
Ndataset2[which((HI_flags == 1)|(FZ_flags == 1))] <- NA

for(i in 1:nrow(Meta)){
	Nint <- Ndataset2[,i]

   if((length(neighbourlist[[i]]) < nstat)|(length(which(is.na(Nint)==F)) < 1)){SOflag <- rep(-1, times=length(Nint))
	}else{

	Nintrain <- rep(0, length=length(Nint))
	Nintrain[which(Nint > 0)] <- 1
	Nintraincum <- cumsum(Nintrain)
	comparestartrowA <- match((Nintraincum-rainyint+1), Nintraincum)-1
	comparestartrowA[which(comparestartrowA == 0)] <- NA
	comparestartrowB <- c(rep(NA, times=compareint-1), 1:(length(Nint)-compareint+1))
	comparestartrow <- ifelse(is.na(comparestartrowB), NA, ifelse((comparestartrowA < comparestartrowB), comparestartrowA, comparestartrowB))

   NeighbourVal <- Ndataset2[,which(Meta$id %in% neighbourlist[[i]])]
   NeighbourVal[which(is.na(Nint)),] <- NA

   cortable <- biastable <-  matrix(NA, ncol=ncol(NeighbourVal), nrow=nrow(NeighbourVal))

   	for(t in 1:length(Nint)){
		if(is.na(comparestartrow[t])){next}
		NeighbourValselec <- NeighbourVal[comparestartrow[t]:t,]
		columnselec <- which(apply(NeighbourValselec, 2, function(x) length(which(is.na(x)==F))) > matchint)
		if(length(columnselec) < nstat){next}
		cortable[t,columnselec] <- apply(NeighbourValselec[,columnselec, drop=FALSE], 2, function(x) cor(x, Nint[comparestartrow[t]:t], use='complete.obs'))
		biastable[t,columnselec] <- apply(NeighbourValselec[,columnselec, drop=FALSE], 2, function(x) mean(Nint[comparestartrow[t]:t]/defaultbiascorrection - x, na.rm=T)/mean(x, na.rm=T) )
   	}

   SOflag <- rep(0, times=length(Nint))
   SOflag[which(apply(cortable, 1, function(x) median(x, na.rm=T)) < corthres)] <- 1
   SOflag[which(apply(cortable, 1, function(x) length(which(is.na(x)==F))) < nstat)] <- -1
	}
   if(exists("SO_flags")==F){ SO_flags <- SOflag
	}else{ SO_flags <- cbind(SO_flags, SOflag) }

   biascorrectiontimeline <- rep(defaultbiascorrection, times=length(Nint))
   if(length(which(SOflag == 0)) > 0){
   biasmed <- apply(biastable, 1, function(x) median(x, na.rm=T))

   for(brow in which(SOflag == 0)){
	biasprev <- biascorrectiontimeline[brow]
	biasnew <- 1 / (1+biasmed[brow])
	if( abs(log(biasnew / biasprev)) > log(1+biasthres) ){
		biascorrectiontimeline[(brow+1):length(biascorrectiontimeline)] <- biasnew }
   }
   }

   if(exists("biascorrectiontable")==F){ biascorrectiontable <- biascorrectiontimeline
	}else{ biascorrectiontable <- cbind(biascorrectiontable, biascorrectiontimeline) }

}
SO_flags <- matrix(SO_flags, ncol=nrow(Meta))
biascorrectiontable <- matrix(biascorrectiontable, ncol=nrow(Meta))
write.csv(SO_flags, file.path(dir, 'SO_flags.csv'), row.names=FALSE)
write.csv(biascorrectiontable, file.path(dir, 'BCF.csv'), row.names=FALSE)
cat('done\n')
