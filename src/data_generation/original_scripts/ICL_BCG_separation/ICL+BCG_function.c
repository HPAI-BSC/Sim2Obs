{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #ifdef SEARCH_CD\
#define MINCDSTARS 300		/* less than this no hope to fit velocity disp. */\
#define MAXCDCYCLE 30\
#define NCLASS 100		/* for velocity histogram */\
void unbind_cD(int lev, int head, int len, int gr, int *SDCLen, int *cDLen)\
\{\
  energy_dat *subpart_egys;\
  int i, j, p, num, part_index, minindex;\
  int max_remove_per_step, unbound, dm_part;\
  double s[3], dx[3], v[3], dv[3];\
  double sqa, H_of_a, pot, minpot = 0;\
  double boxsize, boxhalf;\
  double TotMass, TotMasscd, TotMassST;\
\
  double VelDispDM, VelDispSDC, dv2[3] = \{ 0., 0., 0. \};\
  double VelDispcd, VelDispStars, dv2cd[3] = \{ 0., 0., 0. \};\
  double Rmax, Rfrac, Rcur, dist, dR, dVD;\
  int cDcycle, cDsign, SubLenOrig, nStars, nDiff, nDiffcd;\
  int nin, nout;\
\
  double Vmin, Vmax, VV, aa, ff;\
  double Sigma_cD, Sigma_SDC, k_cD, k_SDC;\
  float xiden;\
  int iden, nclass, vdsearch;\
  double histvel[NCLASS], histsdc[NCLASS];\
  double *BindingEnergyOrig;\
\
  FILE *fp, *f, *f1;\
  FILE *fc1, *fc2, *fv1, *fv2;\
  char bfr[256];\
  int ii;\
\
  *cDLen = len;\
  *SDCLen = 0;\
\
  /* to remove center of mass velocity of the components */\
  double cd_ms_vel[3] = \{ 0., 0., 0. \}, sdc_ms_vel[3] = \{ 0., 0., 0. \};\
\
  boxsize = BoxSize;\
  boxhalf = 0.5 * BoxSize;\
  for(i = 0; i < NCLASS; i++)\
    histvel[i] = 0.0;\
\
  sqa = sqrt(Time);\
\
  H_of_a = Omega0 / (Time * Time * Time) + (1 - Omega0 - OmegaLambda) / (Time * Time)\
#ifdef DARKENERGY\
    + DarkEnergy_a(Time);\
#else\
    + OmegaLambda;\
#endif\
  H_of_a = Hubble * sqrt(H_of_a);\
\
  subpart_egys = mymalloc(len * sizeof(energy_dat));\
  BindingEnergyOrig = mymalloc(len * sizeof(double));\
\
  SubLenOrig = GrP[head].SubLen;	/* they are all the same at this point */\
  /* Check better why last line is not enough, ask ewald ... \
  SubLenOrig = 0;\
  int cur_part = head;\
  int num_part = 0;\
  while (SubLenOrig == 0 && num_part < len)\
    \{\
      if (GrP[cur_part].Level == lev) SubLenOrig = GrP[cur_part].SubLen;\
      cur_part = Next[cur_part];\
      ++num_part;\
    \}\
  */\
\
  for(i = 0, p = head; i < len; i++, p = Next[p])\
    if(GrP[p].Level == lev)\
      \{\
	BindingEnergyOrig[i] = GrP[p].BindingEnergy;\
	GrP[p].SubLen = 0;\
      \}\
  tree_treebuild(lev, head, len);\
\
  /* let's compute the potential energy */\
\
  for(i = 0, p = head, num = 0; i < len; i++, p = Next[p])\
    \{\
      if(GrP[p].Level == lev)\
	\{\
	  part_index = GrP[p].PartIndex;\
\
	  pot = tree_treeevaluate_potential(part_index);\
\
	  /* note: add self-energy */\
	  GrP[p].Potential = pot + P[part_index].mass / Softening;\
	  GrP[p].Potential *= G / Time;\
\
	  num++;\
	\}\
    \}\
\
  s[0] = s[1] = s[2] = v[0] = v[1] = v[2] = 0;\
\
  for(i = 0, p = head, num = 0, TotMass = 0.0; i < len; i++, p = Next[p])\
    \{\
      if(GrP[p].Level == lev)\
	\{\
	  num++;\
	  part_index = GrP[p].PartIndex;\
	  TotMass += P[part_index].mass;\
	  for(j = 0; j < 3; j++)\
	    v[j] += P[part_index].Vel[j] * P[part_index].mass;\
	\}\
    \}\
\
\
  if(num)\
    for(j = 0; j < 3; j++)\
      v[j] /= TotMass;\
\
  /* search vmin, max */\
  Vmin = 1.e33;\
  Vmax = -Vmin;\
  for(i = 0, p = head, num = 0, TotMass = 0.0, nStars = 0; i < len; i++, p = Next[p])\
    \{\
      if(GrP[p].Level == lev)\
	\{\
	  part_index = GrP[p].PartIndex;\
	  if(P[part_index].Type == 4)\
	    \{\
	      nStars++;\
	      VV = 0;\
	      for(j = 0; j < 3; j++)\
		VV += (P[part_index].Vel[j] - v[j]) * (P[part_index].Vel[j] - v[j]);\
	      VV = sqrt(VV);\
	      if(VV < Vmin)\
		Vmin = VV;\
	      if(VV > Vmax)\
		Vmax = VV;\
	    \}\
	\}\
    \}\
\
  if(nStars > 10000)\
    nclass = NCLASS;\
  else if(nStars > 5000)\
    nclass = NCLASS * 0.6;\
  else if(nStars > 1000)\
    nclass = NCLASS * 0.4;\
  else\
    nclass = NCLASS * 0.2;\
\
\
  aa = -nclass * Vmin / (Vmax - Vmin);\
  ff = nclass / (Vmax - Vmin);\
#ifdef DEBUG\
  fprintf(Logfile, " -- STARS %d NCLASS %d  Vmin, Vmax: %f %f\\n", nStars, nclass, Vmin, Vmax);\
#endif\
\
  for(i = 0, p = head, minindex = -1; i < len; i++, p = Next[p])\
    \{\
      if(GrP[p].Level == lev)\
	if(GrP[p].Potential < minpot || minindex == -1)\
	  \{\
	    minpot = GrP[p].Potential;\
	    minindex = p;\
	  \}\
    \}\
\
  for(j = 0; j < 3; j++)\
    s[j] = P[GrP[minindex].PartIndex].Pos[j];	/* position of minimum potential */\
\
#ifdef OUTASCII\
  sprintf(bfr, "%s/postproc_%03d/gal.dat", OutputDir, SnapshotNum);\
  f = fopen(bfr, "w");\
  sprintf(bfr, "%s/postproc_%03d/dm.dat", OutputDir, SnapshotNum);\
  f1 = fopen(bfr, "w");\
#endif\
  for(j = 0; j < 3; j++)\
    dv2[j] = dv2cd[j] = 0.0;\
  for(i = 0, p = head, num = 0, TotMass = TotMassST = 0.0, Rmax = 0.0; i < len; i++, p = Next[p])\
    \{\
      if(GrP[p].Level == lev)\
	\{\
	  part_index = GrP[p].PartIndex;\
	  dR = sqrt(pow(P[GrP[p].PartIndex].Pos[0] - s[0], 2.0) +\
		    pow(P[GrP[p].PartIndex].Pos[1] - s[1], 2.0) +\
		    pow(P[GrP[p].PartIndex].Pos[2] - s[2], 2.0));\
	  if(dR > Rmax)\
	    Rmax = dR;\
\
	  VV = 0.;\
	  for(j = 0; j < 3; j++)\
	    \{\
	      dv[j] = sqa * (P[part_index].Vel[j] - v[j]);\
	      if(Periodic != 0)\
		dx[j] = Time * NEAREST(P[part_index].Pos[j] - s[j]);\
	      else\
		dx[j] = Time * (P[part_index].Pos[j] - s[j]);\
	      dv[j] += H_of_a * dx[j];\
	      if(P[part_index].Type == 1)\
		\{		/* tot veldisp of dm */\
		  dv2[j] += pow(sqa * P[part_index].Vel[j], 2.0) * P[part_index].mass;\
		\}\
	      else if(P[part_index].Type == 4)\
		\{		/* tot veldisp of stars */\
		  dv2cd[j] += pow(sqa * P[part_index].Vel[j], 2.0) * P[part_index].mass;\
		  VV += (P[part_index].Vel[j] - v[j]) * (P[part_index].Vel[j] - v[j]);\
		\}\
	    \}\
\
	  if(P[part_index].Type == 1)\
	    TotMass += P[part_index].mass;\
	  if(P[part_index].Type == 4)\
	    \{\
	      TotMassST += P[part_index].mass;\
	      /* build velocity histogram */\
	      VV = sqrt(VV);\
	      xiden = VV * ff + aa;\
	      iden = (int) (xiden + 0.000001);\
	      if(iden >= nclass)\
		iden = nclass - 1;\
	      if(iden < 0)\
		iden = 0;\
	      histvel[iden] += P[part_index].mass;\
	    \}\
\
	  GrP[p].BindingEnergy = GrP[p].Potential + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);\
	  if(P[part_index].Type == 0)\
	    GrP[p].BindingEnergy += P[part_index].u;\
	  subpart_egys[num].energy = GrP[p].BindingEnergy;\
	  subpart_egys[num].ind = p;\
#ifdef OUTASCII\
	  if(P[part_index].Type == 4)\
	    \{\
	      fprintf(f, "%f %f %f  %d %d\\n",\
		      P[part_index].Pos[0], P[part_index].Pos[1],\
		      P[part_index].Pos[2], GrP[p].Level, GrP[p].SubLen);\
	    \}\
	  if(P[part_index].Type == 1)\
	    \{\
	      fprintf(f1, "%f %f %f  %d %d %e\\n",\
		      P[part_index].Pos[0], P[part_index].Pos[1],\
		      P[part_index].Pos[2], GrP[p].Level, GrP[p].SubLen, GrP[p].BindingEnergy);\
	    \}\
#endif\
	  num++;\
	\}\
    \}\
#ifdef OUTASCII\
  fprintf(f, "# %f %f %f\\n", s[0], s[1], s[2]);\
  fclose(f1);\
  fclose(f);\
#endif\
#ifdef DEBUG\
  fprintf(Logfile, "    BULK VEL %f %f %f\\n", v[0], v[1], v[2]);\
#endif\
\
\
  for(j = 0; j < 3; j++)\
    dv2[j] /= TotMass;\
  VelDispDM =\
    sqrt((1. / 3.) * fabs((dv2[0] - v[0] * v[0]) + (dv2[1] - v[1] * v[1]) + (dv2[2] - v[2] * v[2])));\
  for(j = 0; j < 3; j++)\
    dv2cd[j] /= TotMassST;\
  VelDispStars =\
    sqrt((1. / 3.) * fabs((dv2cd[0] - v[0] * v[0]) + (dv2cd[1] - v[1] * v[1]) + (dv2cd[2] - v[2] * v[2])));\
  for(j = 0; j < nclass; j++)\
    histvel[j] *= ((double) nStars) / TotMassST;\
\
  if(nStars <= MINCDSTARS)\
    \{				/* not enough stars in this group -> no cD */\
      sprintf(bfr, "%s/postproc_%03d/%s", OutputDir, SnapshotNum, CDFILE);\
      fp = fopen(bfr, "ab");\
      fwrite(&gr, sizeof(int), 1, fp);	/* need an info on the fof grp */\
      fwrite(&nStars, sizeof(int), 1, fp);\
      vdsearch = 0;\
      fwrite(&vdsearch, sizeof(int), 1, fp);	/* kind of match */\
      fwrite(&VelDispStars, sizeof(double), 1, fp);	/* evaluated star v.d. */\
      Sigma_cD = Sigma_SDC = k_cD = k_SDC = 0.0;\
      fwrite(&Sigma_cD, sizeof(double), 1, fp);	/* fit values from 2mxw */\
      fwrite(&Sigma_SDC, sizeof(double), 1, fp);	/* if 0, not fitted */\
      fwrite(&k_cD, sizeof(double), 1, fp);\
      fwrite(&k_SDC, sizeof(double), 1, fp);\
      VelDispcd = VelDispSDC = 0.0;\
      fwrite(&VelDispcd, sizeof(double), 1, fp);	/* obtained values from decomp */\
      fwrite(&VelDispSDC, sizeof(double), 1, fp);\
      Rcur = 0.0;\
      fwrite(&Rcur, sizeof(double), 1, fp);	/* used values for bind pot */\
      num = 0;\
      fwrite(&num, sizeof(int), 1, fp);\
      fclose(fp);\
\
      for(i = 0, p = head; i < len; i++, p = Next[p])	/*reset original whole clus */\
	if(GrP[p].Level == lev)\
	  \{\
	    GrP[p].BindingEnergy = BindingEnergyOrig[i];\
	    GrP[p].SubLen = SubLenOrig;\
	  \}\
\
      myfree(BindingEnergyOrig);\
      myfree(subpart_egys);\
      return;\
    \}\
\
  /* here I fit the velocity distribution with TWO maxwellians\
     and use the larger Sigma to match the SDC */\
  Fit2Maxw(Vmin, Vmax, VelDispStars, nclass, histvel, &Sigma_cD, &Sigma_SDC, &k_cD, &k_SDC);\
\
#ifdef DEBUG\
  fprintf(Logfile, "----- gr n. %d  level: %d nstar: %d ntot: %d\\n", gr, lev, nStars, num);\
  fprintf(Logfile, " Vel disp DM: %f   vel disp STARS (ALL): %f\\n", VelDispDM, VelDispStars);\
  fprintf(Logfile, " Mass: DM: %e  STARS: %e\\n", TotMass, TotMassST);\
#endif\
\
\
  cDcycle = 0;\
  cDsign = -1;\
  vdsearch = 1;			/*match SDC; if fails, it will try to match cD */\
\
  Rfrac = 0.21;			/* star with a big R */\
  dR = 0.1;\
  do\
    \{\
      cDcycle++;\
      Rcur = Rmax * Rfrac;	/* use all STAR and only NON-STAR particle inside for the potential */\
#ifdef DEBUG\
      fprintf(Logfile, "\\n >>> cD search cycle %d: dR: %f Rcur: %f * %f = %f\\n",\
	      cDcycle, dR, Rfrac, Rmax, Rcur);\
#endif\
\
      for(i = 0, p = head, num = 0, nin = 0, nout = 0; i < len; i++, p = Next[p])\
	\{\
	  if(GrP[p].Level == lev)\
	    \{\
	      part_index = GrP[p].PartIndex;\
	      dist = sqrt(pow(P[part_index].Pos[0] - s[0], 2.0) +\
			  pow(P[part_index].Pos[1] - s[1], 2.0) + pow(P[part_index].Pos[2] - s[2], 2.0));\
\
	      if(dist > Rcur)\
		\{		/* ALL STARS INCLUDED! */\
		  GrP[p].SubLen = SubLenOrig;\
		  if(P[part_index].Type == 4)\
		    nout++;\
		\}\
	      else\
		\{\
		  GrP[p].SubLen = 0;\
		  if(P[part_index].Type == 4)\
		    nin++;\
		\}\
	    \}\
	\}\
\
      tree_treebuild(lev, head, len);	/* build the tree with only particles inside Rcur */\
\
\
      for(i = 0, p = head, num = 0; i < len; i++, p = Next[p])\
	\{\
	  if(GrP[p].Level == lev)\
	    \{\
\
	      part_index = GrP[p].PartIndex;\
	      pot = tree_treeevaluate_potential(part_index);\
\
	      /* note: add self-energy */\
	      GrP[p].Potential = pot + P[part_index].mass / Softening;\
	      GrP[p].Potential *= G / Time;\
	      num++;\
	    \}\
	\}\
\
\
      for(i = 0, p = head, num = 0; i < len; i++, p = Next[p])\
	\{\
	  if(GrP[p].Level == lev)\
	    \{\
	      part_index = GrP[p].PartIndex;\
	      for(j = 0; j < 3; j++)\
		\{\
		  dv[j] = sqa * (P[part_index].Vel[j] - v[j]);\
		  if(Periodic != 0)\
		    dx[j] = Time * NEAREST(P[part_index].Pos[j] - s[j]);\
		  else\
		    dx[j] = Time * (P[part_index].Pos[j] - s[j]);\
		  dv[j] += H_of_a * dx[j];\
		\}\
\
	      GrP[p].BindingEnergy = GrP[p].Potential + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);\
	      if(P[part_index].Type == 0)\
		GrP[p].BindingEnergy += P[part_index].u;\
	      if(GrP[p].BindingEnergy < 0.0)\
		\{\
		  subpart_egys[num].ind = p;\
		  num++;	/* number of bound parts */\
		\}\
	    \}\
	\}\
\
\
#ifdef OUTASCII\
      sprintf(bfr, "%s/postproc_%03d/sdc_%d.dat", OutputDir, SnapshotNum, gr);\
      fc1 = fopen(bfr, "w");\
      sprintf(bfr, "%s/postproc_%03d/cd_%d.dat", OutputDir, SnapshotNum, gr);\
      fc2 = fopen(bfr, "w");\
      sprintf(bfr, "%s/postproc_%03d/velhist_sdc_%d.dat", OutputDir, SnapshotNum, gr);\
      fv1 = fopen(bfr, "w");\
      sprintf(bfr, "%s/postproc_%03d/velhist_cd_%d.dat", OutputDir, SnapshotNum, gr);\
      fv2 = fopen(bfr, "w");\
#endif\
      /* find out veldisp of diffuse stellar component */\
      for(i = 0; i < NCLASS; i++)\
	histvel[i] = histsdc[i] = 0.0;\
\
      for(j = 0; j < 3; j++)\
	\{\
	   dv2[j] = dv2cd[j] = 0.0;\
	   /* KD new line */\
	   cd_ms_vel[j] = 0.0;\
	   sdc_ms_vel[j] = 0.0;\
	\}\
      for(i = 0, p = head, TotMass = TotMasscd = 0.0, nDiff = nDiffcd = 0; i < len; i++, p = Next[p])\
	\{\
	  part_index = GrP[p].PartIndex;\
	  if(P[part_index].Type != 4)\
	    continue;\
	  VV = 0.;\
	  if(GrP[p].Level == lev && GrP[p].BindingEnergy > 0.0)\
	    \{\
#ifdef OUTASCII\
	      fprintf(fc1, "%f %f %f  %f %f %f   %e  %e %e\\n",\
		      P[part_index].Pos[0], P[part_index].Pos[1], P[part_index].Pos[2],\
		      P[part_index].Vel[0], P[part_index].Vel[1], P[part_index].Vel[2],\
		      P[part_index].mass, GrP[p].Potential, GrP[p].BindingEnergy);\
#endif\
	      TotMass += P[part_index].mass;\
	      nDiff++;\
	      for(j = 0; j < 3; j++)\
		\{\
		  VV += (P[part_index].Vel[j] - v[j]) * (P[part_index].Vel[j] - v[j]);\
		  if(Periodic != 0)\
		    dx[j] = Time * NEAREST(P[part_index].Pos[j] - s[j]);\
		  else\
		    dx[j] = Time * (P[part_index].Pos[j] - s[j]);\
		  dv2[j] += pow(sqa * P[part_index].Vel[j], 2.0) * P[part_index].mass;\
		  /* KD new line */\
		  sdc_ms_vel[j] += pow(P[part_index].Vel[j] - v[j], 2.0) * P[part_index].mass;\
		\}\
	      VV = sqrt(VV);\
	      xiden = VV * ff + aa;\
	      iden = (int) (xiden + 0.000001);\
	      if(iden >= nclass)\
		iden = nclass - 1;\
	      if(iden < 0)\
		iden = 0;\
	      histsdc[iden] += P[part_index].mass;\
	    \}\
	  else if(GrP[p].Level == lev && GrP[p].BindingEnergy <= 0.0)\
	    \{\
#ifdef OUTASCII\
	      fprintf(fc2, "%f %f %f  %f %f %f    %e  %e %e\\n",\
		      P[part_index].Pos[0], P[part_index].Pos[1], P[part_index].Pos[2],\
		      P[part_index].Vel[0], P[part_index].Vel[1], P[part_index].Vel[2],\
		      P[part_index].mass, GrP[p].Potential, GrP[p].BindingEnergy);\
#endif\
	      TotMasscd += P[part_index].mass;\
	      nDiffcd++;\
	      for(j = 0; j < 3; j++)\
		\{\
		  VV += (P[part_index].Vel[j] - v[j]) * (P[part_index].Vel[j] - v[j]);\
		  if(Periodic != 0)\
		    dx[j] = Time * NEAREST(P[part_index].Pos[j] - s[j]);\
		  else\
		    dx[j] = Time * (P[part_index].Pos[j] - s[j]);\
		  dv2cd[j] += pow(sqa * P[part_index].Vel[j], 2.0) * P[part_index].mass;\
		  cd_ms_vel[j] += pow(P[part_index].Vel[j] - v[j], 2.0) * P[part_index].mass;\
		\}\
	      VV = sqrt(VV);\
	      xiden = VV * ff + aa;\
	      iden = (int) (xiden + 0.000001);\
	      if(iden >= nclass)\
		iden = nclass - 1;\
	      if(iden < 0)\
		iden = 0;\
	      histvel[iden] += P[part_index].mass;\
	    \}\
	\}\
#ifdef OUTASCII\
      fclose(fc1);\
      fclose(fc2);\
      for(i = 0; i < nclass; i++)\
	\{\
	  VV = i * (Vmax - Vmin) / ((double) nclass) + Vmin;\
	  fprintf(fv1, "%f %f\\n", VV, histsdc[i] * ((double) nStars) / TotMassST);\
	  fprintf(fv2, "%f %f\\n", VV, histvel[i] * ((double) nStars) / TotMassST);\
	\}\
      fclose(fv1);\
      fclose(fv2);\
#endif\
\
#ifdef DEBUG\
      fprintf(Logfile, "      ---- %f %f %f   %f   %d %d\\n", dv2[0], dv2[1], dv2[2], TotMass, num, nDiff);\
      fprintf(Logfile, "      ---- %f %f %f   %f   %d %d\\n", dv2cd[0], dv2cd[1], dv2cd[2], TotMasscd, num,\
	      nDiffcd);\
      fprintf(Logfile, " Rmax: %f Rcur: %f      inner: %d outer: %d\\n", Rmax, Rcur, nin, nout);\
#endif\
\
      if(TotMass > 0.0)\
	\{\
	  for(j = 0; j < nclass; j++)\
	    histsdc[j] *= ((double) nStars) / TotMass;\
\
	  for(j = 0; j < 3; j++)\
	    \{\
	      dv2[j] /= TotMass;\
	      sdc_ms_vel[j] /= TotMass;\
	    \}\
\
	  /* here I fit the velocity distribution with a maxwellian Sigma to match the SDC */\
	  VelDispSDC = sqrt((1. / 3.) * (sdc_ms_vel[0] + sdc_ms_vel[1] + sdc_ms_vel[2]));\
	  /*\
	  if(nDiff > MINCDSTARS/15)\
	  */\
	    FitMaxw(Vmin, Vmax, VelDispSDC, nclass, histsdc, &VelDispSDC, &k_SDC);\
	  /* KD new line\
	  VelDispSDC = sqrt((1. / 3.) * (fabs(dv2[0] - v[0] * v[0]) +\
					 fabs(dv2[1] - v[1] * v[1]) + fabs(dv2[2] - v[2] * v[2])));\
	  */\
	\}\
      else\
	\{			/* no SDC stars outside rcur, set to high number */\
	  VelDispSDC = 0.;\
	\}\
      if(TotMasscd > 0.0)\
	\{\
	  for(j = 0; j < nclass; j++)\
	    histvel[j] *= ((double) nStars) / TotMasscd;\
\
	  for(j = 0; j < 3; j++)\
	    \{\
	      dv2cd[j] /= TotMasscd;\
	      cd_ms_vel[j] /= TotMasscd;\
	    \}\
          /* here I fit the velocity distribution with a maxwellian Sigma to match the SDC */\
	  VelDispcd = sqrt((1. / 3.) * (cd_ms_vel[0] + cd_ms_vel[1] + cd_ms_vel[2]));\
	  /*\
          if(nDiffcd > MINCDSTARS/15)\
	  */\
	    FitMaxw(Vmin, Vmax, VelDispcd, nclass, histvel, &VelDispcd, &k_cD);\
	  /* KD new line\
	  VelDispcd = sqrt((1. / 3.) * (fabs(dv2cd[0] - v[0] * v[0]) +\
					fabs(dv2cd[1] - v[1] * v[1]) + fabs(dv2cd[2] - v[2] * v[2])));\
	  */\
	\}\
      else\
	\{			/* no cD stars inside Rcur, set to 0. */\
	  VelDispcd = 0.0;\
	\}\
\
#ifdef DEBUG\
      fprintf(Logfile, "   DM vel. disp: %f  cD vel. disp: %f SDC vel. disp: %f\\n",\
	      VelDispDM, VelDispcd, VelDispSDC);\
      fprintf(Logfile, "   fit: Sigma_cD: %f    Sigma_SDC: %f\\n", Sigma_cD, Sigma_SDC);\
#endif\
\
      /* newtonian search for match */\
      if(vdsearch == 1 && TotMass <= 0.0 && cDcycle > 10 && VelDispSDC < Sigma_SDC - TOLERANCE * Sigma_SDC)\
	\{\
	  vdsearch = 2;		/* SDC v.d. always too low */\
	  cDcycle = 0;\
	  Rfrac = 0.21;\
	  dR = 0.1;\
#ifdef DEBUG\
	  printf("\\n\\n _____CHANGE: MATCH CD____\\n\\n");\
#endif\
	\}\
      if(vdsearch == 1 && TotMasscd <= 0.0 && cDcycle > 10 && VelDispSDC > Sigma_SDC + TOLERANCE * Sigma_SDC)\
	\{\
	  vdsearch = 2;		/* SDC v.d. always too high */\
	  cDcycle = 0;\
	  Rfrac = 0.21;\
	  dR = 0.1;\
#ifdef DEBUG\
	  printf("\\n\\n _____CHANGE: MATCH CD____\\n\\n");\
#endif\
	\}\
\
\
      /* match DM */\
      /* OLD  dVD = DISPFAC*VelDispDM - VelDispSDC; */\
      switch (vdsearch)\
	\{\
	case 1:\
	default:\
	  /* match fitted SDC */\
	  dVD = Sigma_SDC - VelDispSDC;\
	  break;\
	case 2:\
	  /* match fitted cD */\
	  dVD = Sigma_cD - VelDispcd;\
	  break;\
	\}\
\
      if(cDsign * dVD < 0.0)	/* one step onward and one backward */\
	dR /= 2.0;		/* need to reduce the dimension of the step */\
      if(dVD < 0. && Rfrac <= dR)	/* avoid negative R */\
	dR = Rfrac / 2.0;\
\
\
      if(dVD < 0.)\
	\{			/* vel. disp. of diffuse stellar component is too high. */\
	  Rfrac -= dR;		/* need LOWER potential: we unbind too few particles */\
	  cDsign = -1;\
	\}\
      else if(dVD > 0. && TotMass > 0.)\
	\{\
	  /* vel. disp. is too low: we unbind too many parts */\
	  Rfrac += dR;		/* need bigger sphere! */\
	  cDsign = +1;\
	\}\
      else if(dVD > 0. && TotMass <= 0.)\
	\{			/* NO SDC!! we bind too much */\
	  Rfrac -= dR;\
	  cDsign = -1;\
	\}\
\
\
    \}\
  while(((vdsearch == 1 && fabs(dVD) > TOLERANCE * Sigma_SDC) || (vdsearch == 2 && fabs(dVD) > TOLERANCE * Sigma_cD)) && cDcycle < MAXCDCYCLE);	/* end of veldisp-matching cycle */\
\
\
\
  /* write out an auxiliary file with cD indexes */\
  sprintf(bfr, "%s/postproc_%03d/%s", OutputDir, SnapshotNum, CDFILE);\
  fp = fopen(bfr, "ab");\
  fwrite(&gr, sizeof(int), 1, fp);	/* fof grp (in order of P-FoF) */\
\
  fwrite(&nStars, sizeof(int), 1, fp);\
  if(cDcycle >= MAXCDCYCLE)\
    \{				/* no decomposition found */\
      /* \
	 KD: remember at least the values ... \
	 VelDispcd = VelDispSDC = 0.0;\
      */\
      vdsearch = 3;\
    \}\
  /* KD bit encode the cycles needed */\
  vdsearch += 8*cDcycle;\
  fwrite(&vdsearch, sizeof(int), 1, fp);	/* kind of match */\
  fwrite(&VelDispStars, sizeof(double), 1, fp);	/* evaluated star v.d. */\
\
  fwrite(&Sigma_cD, sizeof(double), 1, fp);	/* fit values from 2mxw */\
  fwrite(&Sigma_SDC, sizeof(double), 1, fp);\
  fwrite(&k_cD, sizeof(double), 1, fp);\
  fwrite(&k_SDC, sizeof(double), 1, fp);\
  fwrite(&VelDispcd, sizeof(double), 1, fp);	/* obtained values from decomp */\
  fwrite(&VelDispSDC, sizeof(double), 1, fp);\
\
  fwrite(&Rcur, sizeof(double), 1, fp);	/* used values for bind pot */\
\
  fwrite(&num, sizeof(int), 1, fp);\
  for(i = 0; i < num; i++)	/* ALL "cD" parts, not only stars */\
    fwrite(&GrP[subpart_egys[i].ind].ID, sizeof(long long), 1, fp);\
\
  fclose(fp);\
\
  *cDLen = num;\
  *SDCLen = len - num;\
\
  for(i = 0, p = head; i < len; i++, p = Next[p])	/*reset original whole clus */\
    if(GrP[p].Level == lev)\
      \{\
	GrP[p].BindingEnergy = BindingEnergyOrig[i];\
	GrP[p].SubLen = SubLenOrig;\
      \}\
  myfree(BindingEnergyOrig);\
  myfree(subpart_egys);\
\}\
\
#endif}