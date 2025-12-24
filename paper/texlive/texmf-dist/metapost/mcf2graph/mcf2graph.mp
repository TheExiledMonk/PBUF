%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mcf2graph ver 5.29    Copyright (c) 2013-2025   Akira Yamaji
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software
% and associated documentation files (the "Software"), to deal in the Software without restriction,
% including without limitation the rights to use, copy, modify, merge, publish, distribute,
% sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all copies
% or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,EXPRESS OR IMPLIED,
% INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE
% AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%-------------------------------------------------------------------------------------------------
%  mcf2graph is MetaPost macro package convert Molecular Coding Format(MCF) to graphic file
%  sgv/eps/png/mdl molfile
% This package is located at : http://www.ctan.org/pkg/mcf2graph
% Suggestion or request mail to : mcf2graph@gmail.com 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
message "* This is mcf2graph ver 5.29  2025.10.05";
tracingstats:=1; prologues:=3;
%-------------------------------------------------------------------------------------------------
newinternal string EN_,tempc,temps,lines,atoms;
newinternal numeric nA,nB,nC,nD,nE,nL,nN,nP,nR,nS,nU,nX,nY,pX,pY,pB,pc_int;
numeric save_num[],pc_x[][],pc_y[][],param_e[][],pc_cnt[],tbl_atom[],tbl_grp[][],
        tbl_atom_wt[],tbl_atom_mi[],tbl_char_wd[],tbl_char_ht[],at_char[],mc_indent[],error_e[],
        error_b[],unit_lines[],unit_info[],lenw[],mc_x[],mc_y[],mc_b[],error_type[];
string  row[][],save_str[],tbl_atoms[],str_tbl[],arg_s[],mc[],ex[],ad[],op[],tag[],error_s[],
        warning_s[],lib_tag[][],lib_val[][],block_s[],mc,op,ex,ad,libm,No,EN,LN,FM,MW,MI,
        CAT,JN,CAS,USE,LV,mpfont,atomfont,numberfont,forbiddens,blanks,file_input,file_output;
pair    save_pair[],msize,mposition,fsize,fmargin;
boolean bA;
color   rgb_black,rgb_gray,rgb_silver,rgb_white,rgb_blue,rgb_navy,rgb_teal,rgb_green,rgb_lime,
        rgb_aqua,rgb_yellow,rgb_red,rgb_fuchsia,rgb_olive,rgb_purple,rgb_maroon;
%-------------------------------------------------------------------------------------------------
rgb_black=(0,0,0);   rgb_gray=(.5,.5,.5); rgb_silver=(.75,.75,.75); rgb_white=(1,1,1);
rgb_blue=(0,0,1);    rgb_yellow=(1,1,0);  rgb_navy=(1,1,0);         rgb_teal=(0,.5,.5);
rgb_green=(0,.5,0);  rgb_lime=(0,1,0);    rgb_aqua=(0,1,1);         rgb_red=(1,0,0);
rgb_fuchsia=(1,0,1); rgb_olive=(.5,.5,0); rgb_purple=(.5,0,.5);     rgb_maroon=(.5,0,0);
%-------------------------------------------------------------------------------------------------
fig_num:=str_cnt:=tbl_cnt:=mangle:=sw_frame:=sw_trimming:=sw_ext_all:=sw_omit:=0;
sw_output:=sw_numbering:=tag_cnt:=pc_all:=MW_n:=f_expand:=0;
%-------------------------------------------------------------------------------------------------
libm:="main_lib.mcf"; mpfont:="uhvr8r"; atomfont:=numberfont:="draw"; defaultfont:=mpfont;
blanks= "            "; forbiddens=" &()[]{}^=;!'+,`~";
for s="No","EN","JN","LN","MW","MI","FM","CAT","CAS","USE","LV": tag[incr tag_cnt]:=s; endfor
Fig=1; Mcode=2; Info=4; Report=8; MOL2k=16; MOL3k=32; Atom=2; Bond=4; All=8; Group=8; Mol=16;
Outside=1; Inside=2; Bothside:=Outside+Inside;
%-------------------------------------------------------------------------------------------------
BRS=ASCII("{"); BRE=ASCII("}"); BKS=ASCII("["); BKE=ASCII("]"); BSL=ASCII("\"); DQT=ASCII(ditto);
PRS=ASCII("("); PRE=ASCII(")"); QES=ASCII("?"); DLL=ASCII("$"); SMC=ASCII(";"); PCT=ASCII("%");
CMA=ASCII(","); EQU=ASCII("="); AST=ASCII("*"); SLS=ASCII("/"); GTN=ASCII("<"); LTN=ASCII(">");
AMP=ASCII("&"); HSH=ASCII("#"); HAT=ASCII("^"); TLD=ASCII("~"); BQT=ASCII("`"); CLN=ASCII(":");
AMK=ASCII("@"); QUT=ASCII("'"); BAR=ASCII("|"); PLS=ASCII("+"); MIS=ASCII("-"); BLK=ASCII(" ");
%=================================================================================================
let DIV= /; let MUL= *; let PADD = ++; let ef=elseif; def ]]]=] ] ] enddef;
%--default ahangle=45---------------------------------------------------------------------
if ahangle=1:  outputformat:="png"; hppp:=vppp:=0.12; % png format(600dpi)
ef ahangle=2:  outputformat:="png"; hppp:=vppp:=0.06; % png format(1200dpi)
ef ahangle=45: outputformat:="svg"; fi                % svg format  *default
%--default ahlength=4---------------------------------------------------------------------
if ahlength=3: sw_output:=Fig;   f_expand:=1;         % output figure(expanded)
ef ahlength=4: sw_output:=Fig;                        % output figure *default
ef ahlength=5: sw_output:=MOL2k; f_expand:=1;         % output MOL(V2000)
ef ahlength=6: sw_output:=MOL3k; f_expand:=1;         % output MOL(V3000)
ef ahlength=7: sw_output:=Report; fi                  % output report
%-- default bboxmargin=2------------------------------------------------------------------
if bboxmargin=3: ext(defaultfont:=mpfont; defaultscale:=.3; label.rt(EN,(0,0));)
ef bboxmargin=4: sw_output:=Fig+Report;
  ext(defaultfont:=mpfont; defaultscale:=.3;
  label.rt(EN&" / "&MW&" / "&decimal(MW_n-scantokens(MW)),(0,0));) fi
%--default outputtemplate:="%j-%3c."&"svg"------------------------------------------------
if outputformat="svg":                 outputtemplate:="%3c-%{EN_}.svg";
ef outputformat="png": if hppp=0.12:   outputtemplate:="p06_%3c-%{EN_}.png";
                       ef hppp=0.06:   outputtemplate:="p12_%3c-%{EN_}.png"; fi fi
%-----------------------------------------------------------------------------------------
message "* jobname="&jobname;
message "* numbersystem="&numbersystem;
message "* outputformat="&outputformat;
message "* outputtemplate="&outputtemplate;
clearit;
%-------------------------------------------------------------------------------------------------
ratio_chain_ring:=0.66; ratio_atom_bond:=0.36; ratio_thickness_bond:=0.012; ratio_char_atom:=0.12;
ratio_bondgap_bond:=0.15; ratio_hashgap_bond:=0.12; ratio_hash_black:=0.4; ratio_wedge_bond:=0.12;
ratio_atomgap_atom:=0.05; offset_thickness:=0.25; offset_bond_gap:=0.3; offset_hash_gap:=0.1;
offset_atom:=0.8; offset_wedge:=0.2; thickness_frame:=0.2;
max_blength:=10mm; blength:=mangle:=0; max_labelsize:=20mm; dottedline_gap:=1.5;
fsize:=(30mm,20mm); fmargin:=(0.4mm,0.4mm); msize:=(1,1); mposition:=(0.5,0.5);
ahangle:=45; ahlength:=4; defaultsize:=8; defaultscale:=1; labeloffset:=3; ext_defaultline:=0.5;
lonepairdiam:=lonepairspace:=circlediam:=circlepen:=bboxmargin:=0; mc_length:=100;
%-------------------------------------------------------------------------------------------------
def ext(text t)= sw_ext_all:=1; def EXT_ALL = t enddef; enddef;
def ext_clear= sw_ext_all:=0; def EXT_ALL = enddef; enddef;
def wpcs expr n= withpen pencircle scaled n enddef;
def ppcs expr n= pickup pencircle scaled n enddef;
def sbp(expr m,n)expr p=subpath(m*length(p),n*length(p)) of p enddef;
def puts expr s= write s to file_output enddef;
%-------------------------------------------------------------------------------------------------
vardef frac primary n= n-floor n enddef;
vardef iif(expr a,b,c)=if a: b else: c fi enddef;
vardef sfrt(expr a,b,c)= a shifted ((b,0) rotated c) enddef;
vardef fsr(expr n)(expr s)= (substring(0,n-length(s)) of blanks)&s enddef;
vardef fsl(expr n)(expr s)= s&(substring(0,n-length(s)) of blanks) enddef;
vardef fdr(expr n)(expr s)=
  if length(decimal(s))>n: substring (0,n) of decimal(s) else: fsr(n)(decimal(s)) fi enddef;
vardef fdl(expr n)(expr s)= fsl(n)(decimal(s)) enddef;
vardef subc(expr s,n)= substring(n-1,n) of s enddef;
%=================================================================================================
pc_emb_start=1001;     % 1001 => 1900   for embedded pcode (max 900)
pc_emi_start=1901;     % 1901 => 2000   for embedded internal pcode (max 100)
pc_usr_start=2001;     % 2001 => 3000   for user     pcode (max 1000)
pc_int_start=3000;     % 3001 => 4000   for internal pcode (max 1000)
%-------------------------------------------------------------------------------------------------
def def_com(expr n)(text tx)= nA:=n; forsuffixes $=tx: $=nA; nA:=nA+1; endfor enddef;
def_com(-4090)(_com,_jp_atom,_jp_absA,_jp_bond,_cyc,_cyc_sB,_cyc_eB,_set_line,_tmp_line,_tmp_rot,
  _tmp_env,_tmp_len,_chg_len,_get_len,_set_len,_rot_ang,_adj_ang,_chg_env,_set_colorA,_set_colorB,
  _grp_si,_grp_dm,_grp_wf,_grp_zf,_set_adrA,_set_adrB,_mk_bond,_set_atom,_auto_ang,
  _set_charge,_add_charge,_chg_atom,_fuse,_fuse_ext,_size_atom,_adr,_numeric_rot,_jump_at,
  _nop,_mark,_moff,_term,_end,_grp_s,_grp_e,_rest,_ring,_chain,
  hz,vt,lr,rl,si,dl,dl_,dr,dr_,db,dm,dm_,tm,wf,wb,bd,bz,zf,zb,dt,wv,nl,vf,vb,wf_r,wb_r,bd_r,
  arc_lb,arc_br,arc_lbr,arc_ltr,si_,wf_,wb_,zf_,zb_,wv_,bd_);
%-------------------------------------------------------------------------------------------------
def parameter_list=
  sw_numbering,sw_output,sw_ext_all,sw_frame,sw_trimming,sw_omit,ratio_atom_bond,
  ratio_thickness_bond,ratio_chain_ring,ratio_bondgap_bond,ratio_hash_black,ratio_hashgap_bond,
  ratio_wedge_bond,ratio_atomgap_atom,ratio_char_atom,offset_atom,offset_wedge,max_blength,
  offset_hash_gap,offset_bond_gap,thickness_frame,offset_thickness,defaultsize,defaultscale,
  labeloffset,mangle,blength,fsize,fmargin,msize,mposition,defaultfont,atomfont,dottedline_gap
enddef;
%-------------------------------------------------------------------------------------------------
def store_restore_par(expr n)(text t)=
  nA:=nB:=nC:=0;
  if n=0: for $=t: if numeric $: save_num[incr nA]:=$; ef pair $: save_pair[incr nB]:=$;
                   ef string  $: save_str[incr nC]:=$; fi endfor
  ef n=1: for $=t: if numeric $: if save_num[incr nA]<>$:  save_num[nA]:=$; fi
                   ef pair $:    if save_pair[incr nB]<>$: save_pair[nB]:=$; fi
                   ef string $:  if save_str[incr nC]<>$:  save_str[nC]:=$; fi fi endfor
  ef n=2: forsuffixes $=t: if numeric $: if $<>save_num[incr nA]:  $:=save_num[nA]; fi
                           ef pair    $: if $<>save_pair[incr nB]: $:=save_pair[nB]; fi
                           ef string  $: if $<>save_str[incr nC]:  $:=save_str[nC]; fi fi endfor
  fi
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def beginfigm=
  begingroup
  save ','',`,``,@,#,#@,##,|=,|<,**,++,f_ext,blen,ext,add,ang_br,fw_n,bw_n,at_cmm,bond_cntA,
    warning_cnt,hideH,hideH_cnt,cntM,cntA,cntB,minX,minY,maxX,maxY,posA,posM,lineB,
    sB,eB,angB,angA,lenB,angX,numS,sumA,bond_num,wdM,htM,chargeA,addA,add_rot,mol_pic,color_list,
    no,mw,mi,fm,grp_num,f_type,ay,bx,by,pcnt;
  numeric hideH[],lineB[],sB[],eB[],angB[],angA[],lenB[],angX[],strA[],sumA[],bond_num[],
          wdM[],htM[],chargeA[],addA[],add_rot[];
  pair posA[],posM[][];
  string mi,fm,mw;
  picture mol_pic[];
  color color_list[];
  %-----------------------------------------------------------------------------------------------
  store_restore_par(1)(parameter_list);
  mc:=ex:=ad:=op:=mw:=mi:=fm:=""; No:=EN:=JN:=LN:=MW:=FM:=MI:=CAS:=CAT:=LV:=USE:="-";
  f_ext:=cntM:=mc_row:=ex_row:=ad_row:=op_row:=param_int:=warning_cnt:=error_cnt:=0;
  let ext=ext_to_fig; let add=add_to_molecule; let ++=add_to_molecule; let **=ext_to_fig;
  def ' = define_parts enddef; def @ expr p= mposition:=p; enddef; def # expr p= fsize:=p; enddef;
  def #@ expr p=fmargin:=p; enddef; def `(expr s)=define_grp_string(s)() enddef;
  def `` = define_grp_string enddef; def ## expr p = msize:=p; enddef;
  def |=(expr n) = blength:=n; enddef; def |<(expr n) = max_blength:=n; enddef;
  pc_num:=pc_usr_start; pc_int:=pc_int_start; fig_num:=fig_num+1; 
  mol_pic[0]:=nullpicture;
enddef;
%-------------------------------------------------------------------------------------------------
def endfigm=
  if cntM>=1:
    if scan_bit(sw_output,Fig):
    if EN<>"-": EN_:=forbidden_to_underbar(EN); else: EN_:="no_name"; fi
      beginfig(fig_num)
        if sw_ext_all=1: ext_to_fig(EXT_ALL); fi
        if sw_trimming>=1:
          nA:=nC:=4095; nB:=nD:=-4095;
          for i=1 upto cntM: if xpart(posM[1][i])<nA: nA:=xpart(posM[1][i]); fi
                             if xpart(posM[2][i])>nB: nB:=xpart(posM[2][i]); fi
                             if ypart(posM[1][i])<nC: nC:=ypart(posM[1][i]); fi
                             if ypart(posM[2][i])>nD: nD:=ypart(posM[2][i]); fi
          endfor
          fig_wd:=nB-nA+2xpart(fmargin); fig_ht:=nD-nC+2ypart(fmargin); fsize:=(fig_wd,fig_ht);
          for i=1 upto cntM:
            posM[0][i]:=posM[0][i]+fmargin-(nA,nC); posM[1][i]:=posM[1][i]+fmargin-(nA,nC);
          endfor
        fi
        if scan_bit(sw_frame,Outside): draw_frame((0,0),fsize,thickness_frame);
        else: draw_corner(fsize,0.004); fi
        if scan_bit(sw_frame,Inside): draw_frame(fmargin,fsize-2fmargin,thickness_frame); fi
        for i=1 upto cntM:
          addto currentpicture also mol_pic[i] shifted posM[0][i]; mol_pic[i]:=nullpicture;
          if scan_bit(sw_frame,Mol): ext(draw_frame(p[i],(w[i],h[i]),thickness_frame)) fi
        endfor
        if f_ext=1: addto currentpicture also mol_pic[0]; mol_pic[0]:=nullpicture; fi
      endfig;
      clearit;
    fi
  ef scan_bit(sw_output,Fig):
    if EN<>"-": EN_:=forbidden_to_underbar(EN); else: EN_:="no_name"; fi
    beginfig(fig_num)
      defaultscale:=.6; draw_frame((0,0),fsize,thickness_frame) label("no figure",0.5fsize);
    endfig;
    clearit;
  fi
  if scan_bit(sw_output,Mcode):  proc_mc_out; fi
  if scan_bit(sw_output,Report): proc_report_out; fi
  if scan_bit(sw_output,MOL2k):  proc_mol_out(2000); fi
  if scan_bit(sw_output,MOL3k):  proc_mol_out(3000); fi
  store_restore_par(2)(parameter_list);
  message "["&decimal(fig_num)&"]EN:"&EN; message "";
  endgroup; % [;]
enddef;
%=================================================================================================
def readm(text t)= for $=t: mc[incr mc_row]:=$; mc:=mc&mc[mc_row]; endfor enddef;
def getm(expr a)= if string a: read_unit(get_adr("EN",a)) ef numeric a: read_unit(a) fi enddef;
%-------------------------------------------------------------------------------------------------
def read_unit(expr n)=
  f_line:=0;
  if (n>=1)and(n<=ucount):
    for i=1 upto unit_info[n]: for j=1 upto tag_cnt:
      if lib_tag[n][i]=tag[j]: scantokens(tag[j]):=lib_val[n][i]; fi
    endfor endfor
    for i=1 upto unit_lines[n]:
      lines:=row[n][i]; firsta:=ASCII(lines); exitif firsta=SMC;
      if firsta=CLN: f_line:=1; ef firsta=EQU: f_line:=2; ef firsta=AST: f_line:=3;
      ef firsta=PLS: f_line:=4; ef firsta=PCT:
      else: fw_n:=scan_char(" ",lines,1,1)-1; temps:=substring(fw_n,length(lines)) of lines;
            if f_line=1: mc_indent[incr mc_row]:=fw_n; mc[mc_row]:=temps; mc:=mc&temps;
            ef f_line=2: op[incr op_row]:=temps; op:=op&temps;
            ef f_line=3: ex[incr ex_row]:=temps; ex:=ex&temps;
            ef f_line=4: ad[incr ad_row]:=temps; ad:=ad&temps; fi fi
    endfor
  fi 
enddef;
%-------------------------------------------------------------------------------------------------
vardef get_adr(expr t,v)=
  save adr_n; adr_n:=0;
  for i=1 upto ucount: for j=1 upto unit_info[i]:
    if (lib_tag[i][j]=t)and(lib_val[i][j]=v): adr_n:=i; fi exitif adr_n>=1;
  endfor endfor adr_n
enddef;
%-------------------------------------------------------------------------------------------------
def putm= if op_row>=1: scantokens(op) fi
          if mc_row>=1: drawm(mc) fi
          if ad_row>=1: add(scantokens(ad)) fi
          if ex_row>=1: ext(scantokens(ex)) fi
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef pc_extend(expr p,a,b,c)=
  param_e[incr param_int][1]:=a; param_e[param_int][2]:=b; param_e[param_int][3]:=c; (p,param_int)
enddef;
%-------------------------------------------------------------------------------------------------
error_s[1]="*Lack argument  "; error_s[2]="*Extra argument "; error_s[3]="*Forbidden char ";
error_s[4]="*Invalid adrA   "; error_s[5]="*Invalid adrB   "; error_s[6]="*Unknown found  ";
error_s[7]="*Not match ( )  ";
def message_error(expr n,e) = error_b[incr error_cnt]:=n; error_e[error_cnt]:=e;
                              message "["&decimal(fig_num)&"]"&error_s[e]; message ""; enddef;
%-------------------------------------------------------------------------------------------------
vardef abs_adress primary n = if numeric n: n-4095 else: _nop fi enddef;
primarydef a from_until b = a for i=a+1 upto b: 'i endfor enddef;
primarydef a bop_qut b = if (known a)and(known b): add_parts(a,b) else: _nop fi enddef;
primarydef a fuse__ b = pc_extend(_fuse_ext,a,b,2) enddef;
primarydef a fuse___ b = pc_extend(_fuse_ext,a,b,3) enddef;
primarydef a fuse____ b = pc_extend(_fuse_ext,a,b,4) enddef;
tertiarydef a bop_equ b = if (known a)and(known b):
  if (numeric a)or(pair a): set_bond(a,b) else: _nop fi else: _nop fi enddef;
tertiarydef a bop_col b =  if (known a)and(known b):
  if (numeric a)or(pair a): set_atom(a,b) else: _nop fi else: _nop fi enddef;
tertiarydef a bop_hat b= if numeric a: if a<pc_emb_start:(_tmp_rot,b),a else:(_tmp_rot,b)'a fi
  else: (_tmp_rot,b)'a fi enddef;
tertiarydef a bop_til b= if numeric a:if a<pc_emb_start:(_tmp_line,b),a else:(_tmp_line,b)'a fi
  else: (_tmp_line,b)'a fi enddef;
tertiarydef a bop_bqu b= if numeric a: if a<pc_emb_start:(_tmp_len,b),a else:(_tmp_len,b)'a fi
  else: (_tmp_len,b)'a fi enddef;
tertiarydef a bop_lth b= if numeric a: if a<pc_emb_start: (_tmp_env,b),a
  ef (a>=!2)and(a<=!20): b for i=0 upto a-!2:,b endfor else: (_tmp_env,b)'a fi
  else: (_tmp_env,b)'a fi enddef;
def move_pos(expr n) =
  <<$0,angle(pc_x[n][1],pc_x[n][2])~nl`length(pc_x[n][1],pc_x[n][2]),<<$0 enddef;
%-------------------------------------------------------------------------------------------------
def set_length primary n = if numeric n: (_chg_len,n) else: _nop fi enddef;
def rot_angle  primary n = if numeric n: (_rot_ang,n) else: _nop fi enddef;
def cyc_atom   primary n = if numeric n: (_cyc,n)     else: _nop fi enddef;
def jump_atom secondary n = if n>=pc_int_start: move_pos(n) else: (_jp_atom,n) fi enddef;
def jump_atom_abs secondary n = if n>=pc_int_start: @.$1,move_pos(n) else: @.$n fi enddef;
def grp_si secondary n = if numeric n: (_grp_si,n) else: _nop fi enddef;
def grp_dm secondary n = if numeric n: (_grp_dm,n) else: _nop fi enddef;
def grp_wf secondary n = if numeric n: (_grp_wf,n) else: _nop fi enddef;
def grp_zf secondary n = if numeric n: (_grp_zf,n) else: _nop fi enddef;
%-------------------------------------------------------------------------------------------------
def redefine_tokens=
  save /,//,*,/*,*/,*/*,~,^,`,',<<,>>,==,::,@,@$,$,@@,&&,&$,#,##,|,||,--,---,----,_,
       CP,CA,MB,CB,CC,DL;
  | :=mark_adress; || :=reset_adress; ##:=reset_length; let '=bop_qut; let == =bop_equ;
  let :: =bop_col; let ^=bop_hat; let ~=bop_til; let >> =bop_lth; let `=bop_bqu; let --=fuse__;
  let --- =fuse___; let ---- =fuse____; let << =rot_angle; let @ =jump_atom; let && =cyc_atom;
  let #=set_length; let /=grp_si; let //=grp_dm; let */=grp_wf; let /*=grp_zf;
  let CP=pc_add; let CA=pc_add_adrA; let CB=pc_add_adrB; let _=from_until;
  def */* expr p=/p~wv enddef; def @@ =read_adr enddef; def * primary n =(n+0.1) enddef;
  def DL(expr p)=CP(_set_adrB,p) CP(_set_line,dl) enddef; def MB(expr p)=CP(_mk_bond,p) enddef;
  def CC(expr p)=CP(_com,p) enddef; def $ =abs_adress enddef; def &&$=&&.$ enddef;
  def @$ =jump_atom_abs enddef;
enddef;
%=================================================================================================
def read_mcf_string(expr s)=
  begingroup
  save len_s,at_eqcl,prs_cnt,at_prs,at_prs,at_temp,f_temp,eqcl_cnt,f_binop;
  numeric at_eqcl[],at_prs[],f_binop[];
  block_cnt:=split_str(s,",")(block_s); block_s[incr block_cnt]:="(_com,_term)";
  %------------------------------------------------------------------------------
  for i=1 upto block_cnt:
    error_type[i]:=check_block(block_s[i]);
    if error_type[i]=0:
      prs_cnt:=eqcl_cnt:=at_temp:=f_temp:=0; temps:=block_s[i]; len_s:=length(temps);
      for j=1 upto len_s:
        tempc:=substring(j-1,j) of temps;
        if (tempc="'")or(tempc="^")or(tempc="`")or(tempc="~")or(tempc=">")or(tempc="_"):f_temp:=1;
        ef tempc="(": at_temp:=j; f_temp:=0;
        ef (tempc=":")or(tempc="="):
          at_eqcl[incr eqcl_cnt]:=j; at_prs[eqcl_cnt]:=at_temp; f_binop[eqcl_cnt]:=f_temp; fi
      endfor
      for j=eqcl_cnt downto 1:
        if f_binop[j]>0: temps:=substring(0,at_prs[j]) of temps&"@@("&
                                substring(at_prs[j],at_eqcl[j]-1) of temps&")"&
                                substring(at_eqcl[j]-1,length(temps)) of temps; fi
      endfor
      if scan_c(":",temps)>0: temps:=replace_str(temps,":","::"); fi
      if scan_c("=",temps)>0: temps:=replace_str(temps,"=","=="); fi
      if scan_c(">",temps)>0: temps:=replace_str(temps,">",">>"); fi
      if scan_c("<",temps)>0: temps:=replace_str(temps,"<","<<"); fi
      if scan_c("&",temps)>0: temps:=replace_str(temps,"&","&&"); fi
      block_s[i]:=temps;
    else:  message_error(i,error_type[i]);
    fi
  endfor
  %-----------------------------------------------------------------------------------
  redefine_tokens;
  mc_cnt:=0;
  for i=1 upto block_cnt:
    if error_type[i]=0:
      for $=scantokens(block_s[i]):
        if known $:
          if numeric $: if $>=pc_emb_start: expand_pcode($,i);
                        ef $=_nop: message_error(i,6);
                        else: mc_x[incr mc_cnt]:=_mk_bond; mc_y[mc_cnt]:=$; mc_b[mc_cnt]:=i; fi
          ef pair $: mc_x[incr mc_cnt]:=xpart($); mc_y[mc_cnt]:=ypart($); mc_b[mc_cnt]:=i; fi
        else: message_error(i,6); fi
      endfor
    fi
  endfor
  endgroup
enddef;
%-------------------------------------------------------------------------------------------------
vardef define_parts(text t)=
  pc_cnt[incr pc_num]:=0;
  redefine_tokens;
  for $=t: if known $: if pair $: pc_x[pc_num][incr pc_cnt[pc_num]]:=xpart($);
                                  pc_y[pc_num][pc_cnt[pc_num]]:=ypart($);
                       ef numeric $: if $>=pc_emb_start: pc_x[pc_num][incr pc_cnt[pc_num]]:=$;
                                     ef $=_nop: message_error(0,6);
                                     else: pc_x[pc_num][incr pc_cnt[pc_num]]:=_mk_bond;
                                           pc_y[pc_num][pc_cnt[pc_num]]:=$; fi fi fi
  endfor pc_num
enddef;
%-------------------------------------------------------------------------------------------------
def expand_pcode(expr a,b)=
  for i=1 upto pc_cnt[a]:
    if pc_x[a][i]>=pc_emb_start: expand_pcode(pc_x[a][i],b);
    else: mc_x[incr mc_cnt]:=pc_x[a][i]; mc_y[mc_cnt]:=pc_y[a][i]; mc_b[mc_cnt]:=b; fi
  endfor
enddef;
%-------------------------------------------------------------------------------------------------
vardef read_adr(text t)=
  save '; let ' = , ;
  pc_int:=pc_int+1; pcnt:=0;
  for $=t:
    if known $: if numeric $:if $<>_nop: pc_x[pc_int][incr pcnt]:=_adr; pc_y[pc_int][pcnt]:=$; fi
                ef pair $: pc_x[pc_int][incr pcnt]:=xpart($); pc_y[pc_int][pcnt]:=ypart($); fi fi
  endfor
  pc_cnt[pc_int]:=pcnt; pc_int
enddef;
%-------------------------------------------------------------------------------------------------
vardef add_parts(expr a,b)=
  if (numeric a)and(numeric b):
    if (a>=pc_int_start)and(b<pc_int_start): pc_x[a][incr pc_cnt[a]]:=b;
    ef (a<pc_emb_start)and(b<pc_emb_start):
      pc_x[incr pc_int][1]:=a; pc_x[pc_int][2]:=b; pc_cnt[pc_int]:=2;
    else: pc_x[incr pc_int][1]:=a; pc_x[pc_int][2]:=b; pc_cnt[pc_int]:=2; fi
  ef (numeric a)and(pair b):
    if a>=pc_int_start: pc_x[a][incr pc_cnt[a]]:=xpart b; pc_y[a][pc_cnt[a]]:=ypart b;
    ef a>=pc_emb_start: pc_x[incr pc_int][1]:=a; pc_x[pc_int][2]:=xpart b;
                           pc_y[pc_int][2]:=ypart b; pc_cnt[pc_int]:=2; fi
  ef (pair a)and(numeric b): pc_x[incr pc_int][1]:=xpart a; pc_y[pc_int][1]:=ypart a;
                             pc_x[pc_int][2]:=b; pc_cnt[pc_int]:=2;
  ef (pair a)and(pair b):
    pc_x[incr pc_int][1]:=xpart a; pc_y[pc_int][1]:=ypart a;
    pc_x[pc_int][2]:=xpart b; pc_y[pc_int][2]:=ypart b; pc_cnt[pc_int]:=2; fi
  pc_int
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fuse_ring(expr a,b) =
  CP(_jp_bond,a) CP(_rot_ang,180) CP(_get_len,a) for i=3 upto b: MB(360 DIV b) endfor
  CP(_set_len,_end) if a<=0: CP(_cyc_eB,a-b+2) else: CP(_cyc_eB,a) fi
enddef;
def fuse_ring_bonds(expr s,e,b,c) =
  CP(_jp_bond,s) CP(_rot_ang,180)
  if b=6: CP(_set_len,1) for i=2 upto c: MB(60) endfor
  ef b=5: if c=2: CP(_set_len,1.25) MB(80) ef c=3: CP(_set_len,1.1) MB(78) MB(72) fi
  ef b=4: CP(_set_len,1.225) MB(105) fi
  CP(_set_len,_end) if e<=0: CP(_cyc_eB,e-c+1) else: CP(_cyc_eB,e) fi
enddef;
def fuse_ring_size(expr a,b,c) =
  CP(_jp_bond,a) CP(_rot_ang,180) CP(_set_len,c)
  if b=5: MB(72-((10c-9) MUL 1.5)) MB(63+10c) MB(63+10c)
  ef b=6: MB(68-10c) for i=1 upto 3: MB(60+(5c-4)) endfor fi
  CP(_set_len,_end) if a<=0: CP(_cyc_eB,a-b+2) else: CP(_cyc_eB,a) fi
enddef;
%=================================================================================================
vardef set_bond(expr a,b) =
  pc_cnt[incr pc_int]:=0;
  if numeric b:
    if (b>=si)and(b<=bd_):
      if a>=pc_int_start:
        for i=1 upto pc_cnt[a]: if pc_x[a][i]=_adr: CP(_set_adrB,pc_y[a][i]) CP(_set_line,b) fi
        endfor
      else: CP(_set_adrB,a) CP(_set_line,b) fi
    ef (b>=?3)and(b<=Ph2)or(b>=pc_int_start):
      if numeric a:
        if a>=pc_int_start:
          for i=1 upto pc_cnt[a]:
            ay:=pc_y[a][i];
            if pc_x[a][i]=_adr: if b=Ph1: fuse_ring(ay,6) DL(-2) DL(-4)
                                   ef b=Ph2: fuse_ring(ay,6) DL(-1) DL(-3) DL(-5)
                                   else: fuse_ring(ay,b-?3+3) fi
            ef pc_x[a][i]=_fuse_ext:
              fuse_ring_bonds(param_e[ay][1],param_e[ay][2],b-?4+4,b-?4+4-param_e[ay][3]) fi
          endfor
        else: if b=Ph1: fuse_ring(a,6) DL(-2) DL(-4)
              ef b=Ph2: fuse_ring(a,6) DL(-1) DL(-3) DL(-5)
              ef b>=pc_int_start: if (pc_x[b][1]=_chg_len)and(pc_x[b][2]>=?5)and(pc_x[b][2]<=?6):
                                    fuse_ring_size(a,pc_x[b][2]-?3+3,pc_y[b][1]) fi
              else: fuse_ring(a,b-?3+3) fi
        fi
      ef pair a:
        if xpart a=_fuse_ext:
          ay:=ypart a; fuse_ring_bonds(param_e[ay][1],param_e[ay][2],b-?4+4,b-?4+4-param_e[ay][3])
        fi
      fi
      else: message_error(block_cnt,6); fi
  ef color b: cntC:=cntC+1; color_list[cntC]:=b; CB(_set_colorB,cntC,a) fi
  pc_int
enddef;
%-------------------------------------------------------------------------------------------------
vardef set_atom(expr a,b)=
  pc_cnt[incr pc_int]:=pcnt:=f_type:=0;
  if numeric b: if (b>=?)and(b<=?dm): bx:=pc_x[b][1]; by:=pc_y[b][1]; f_type:=1; fi
  ef pair b: bx:=xpart(b); by:=ypart(b); f_type:=1;
  ef color b: f_type:=2; fi
  if f_type=0:
    if (b>pc_emb_start)and(b<=pc_atm_end): CA(_chg_atom,b,a)
    ef b=NH: CA(_chg_atom,N,a) if f_expand=0: CP(_tmp_line,nl) fi
             CC(_grp_s) CA(_grp_si,H,a) CC(_grp_e)
    ef b=N?: CA(_chg_atom,N,a) CC(_grp_s) CA(_grp_si,NO_ATOM,a) CC(_grp_e)
    ef b=N?2: CA(_chg_atom,N,a) CC(_grp_s) CA(_grp_si,!,a) CC(_grp_e)
    ef b=??: CP(_tmp_rot, 35) CC(_grp_s) CA(_grp_si,NO_ATOM,a) CC(_grp_e)
             CP(_tmp_rot,-35) CC(_grp_s) CA(_grp_si,NO_ATOM,a) CC(_grp_e)
    ef b=n_: CC(_grp_s) CA(_set_charge,MIS,a) CC(_grp_e)
    ef b=p_: CC(_grp_s) CA(_set_charge,PLS,a) CC(_grp_e)
    ef b=_nop: message_error(block_cnt,6); fi
  ef f_type=1:
    if numeric a:
      CC(_grp_s)
      if a>=pc_emb_start:
        for i=1 upto pc_cnt[a]: if pc_x[a][i]=_adr: CP(_set_adrA,pc_y[a][i]) CP(bx,by)
                                   else: CP(pc_x[a][i],pc_y[a][i]) fi
        endfor
      else: CA(bx,by,a) fi
      CC(_grp_e)
    fi
  ef f_type=2: cntC:=cntC+1; color_list[cntC]:=b; CA(_set_colorA,cntC,a)
  fi
  pc_cnt[pc_int]:=pc_cnt[pc_int]; pc_int
enddef;
%-------------------------------------------------------------------------------------------------
def pc_add(expr x,y)=pc_x[pc_int][incr pc_cnt[pc_int]]:=x;pc_y[pc_int][pc_cnt[pc_int]]:=y; enddef;
def pc_add_adress(expr x,y,n,c)=
  if n>=pc_emb_start:
    for i=1 upto pc_cnt[n]: if pc_x[n][i]=_adr: pc_add(c,pc_y[n][i]) pc_add(x,y)
                            else: pc_add(pc_x[n][i],pc_y[n][i]) fi
    endfor
  else: pc_add(c,n) pc_add(x,y) fi
enddef;
def pc_add_adrA(expr x,y,n)= pc_add_adress(x,y,n,_set_adrA) enddef;
def pc_add_adrB(expr x,y,n)= pc_add_adress(x,y,n,_set_adrB) enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def ext_setup=
  pickup pencircle scaled ext_defaultline;
  dotlabeldiam:=3; labeloffset:=3; em:=defaultscale*defaultsize; defaultfont:=mpfont;
  let # = scaled; let << = rotated; let => = shifted; __ = (1,0); An:=cntA; Bn:=cntB;
  primarydef a /* b = point b of a enddef;
enddef;
%-------------------------------------------------------------------------------------------------
def add_to_molecule(text t)=
  begingroup
  save w,h,n,l,p,am,aw,A,B,An,Bn,plus,minus,lonepair,__,#,=>,<<,/*;
  numeric A[]dir,B[]up,A[]ang,B[]ang;
  pair __,p[],A[],B[]s,B[]e,B[]m,A[]up,A[]left,A[]right,A[]down,B[]up,B[]left,B[]right,B[]down;
  path B[];
  def plus = circled_plus_add enddef; def minus = circled_minus_add enddef;
  def lonepair = lone_pair_add enddef;
  ext_setup;
  w:=mol_wd; h:=mol_ht; l:=blen; aw:=atom_wd; p0:=(minX,minY);
  lonepairdiam:=0.3aw; lonepairspace:=.7aw; circlediam:=.6aw; circlepen:=.2;
  for i=1 upto cntA:
    A[i]:=posA[i]; A[i]ang:=angX[i]; A[i]up:=dir(angX[i]);
    A[i]left:=dir(angX[i]+90); A[i]right:=dir(angX[i]-90); A[i]down:=dir(angX[i]+180);
  endfor
  for i=1 upto cntB:
    B[i]s:=posA[sB[i]]; B[i]e:=posA[eB[i]]; B[i]m:=0.5[B[i]s,B[i]e]; B[i]:=B[i]s--B[i]e;
    B[i]ang:=angB[i]; B[i]up:=dir(angB[i]);
    B[i]down:=dir(angB[i]+180); B[i]left:=dir(angB[i]+90); B[i]right:=dir(angB[i]-90);
  endfor
  t addto mol_pic[cntM] also currentpicture;
  clearit;
  endgroup; % [;]
enddef;
%-------------------------------------------------------------------------------------------------
def ext_to_fig(text t)=
  begingroup
  save w,h,An,Bn,wd,ht,n,p,am,aw,__,#,<<,=>,/*;
  pair __,p[];
  ext_setup;
  w:=xpart(fsize); h:=ypart(fsize);
  w0:=w-2xpart(fmargin); h0:=h-2ypart(fmargin); p0:=fmargin; aw:=atom_wd; n:=cntM;
  for i=1 upto n: p[i]:=posM[1][i]; w[i]:=wdM[i]; h[i]:=htM[i]; endfor
  t addto mol_pic[0] also currentpicture; clearit; f_ext:=1;
  endgroup; % [;]
enddef;
%-------------------------------------------------------------------------------------------------
vardef circled_plus_add= nA:=circlediam; nB:=circlepen;
  image(draw fullcircle scaled nA wpcs nB;
        draw (-.5nA,0)--(.5nA,0) wpcs nB; draw (0,-.5nA)--(0,.5nA) wpcs nB;) enddef;
vardef circled_minus_add= nA:=circlediam; nB:=circlepen;
  image(draw fullcircle scaled nA wpcs nB; draw (-.5nA,0)--(.5nA,0) wpcs nB;) enddef;
vardef lone_pair_add expr n=
  image(draw (0,0) wpcs lonepairdiam; draw ((0,lonepairspace) rotated n) wpcs lonepairdiam;)
enddef;
%=================================================================================================
def drawm(expr s)=
  begingroup
  save temp_strA,temp_cntB,f_bra,f_term,f_at,f_lineT,f_rotT,f_lenT,f_envT,angL,cpos,tpos,factor,
       m_wd,m_ht,raise_pos,slen,sdir,char_wd,char_ht,tcol,f_col,knownA,markA,markB,saveA,saveB,
       lenC,lenL,lenT,lineT,angT,rotT,envT,envB,posBs,posBe,grp_cnt,grp_parts,grp_line,
       grp_len,grp_env,grp_rot,colorA,colorB,save_lenS,save_lenL,spos,slen,
       aW,aH,fW,fH,hW,hW,hH,qH,fP,hP,ww,aw,ap,am,Ls,Le,pA,zA,zL;
  numeric grp_adr[],grp_line[],grp_len[],grp_env[],grp_rot[],grp_parts[],grp_lenL[],
          colorA[],colorB[];
  pair cpos,tpos,spos,raise_pos,posBs,posBe,pA,Ls,Le;
  path frameA[],zA,zL;
  %-----------------------------------------------------------------------------------------------
  if f_expand=1: lenC:=1; else: lenC:=-ratio_chain_ring; fi
  cntA:=cntB:=cntC:=grp_num:=0; str_tbl[0]:="C"; str_cnt:=2000; save_lenL:=save_lenS:=lenC;
  fig_wd:=xpart(fsize); fig_ht:=ypart(fsize);
  %===============================================================================================
  read_mcf_string(s);
  proc_bond_atom(1);
  backboneA:=cntA; backboneB:=cntB;
  if (grp_num>0)and(not scan_bit(sw_omit,Group)): expand_grp(1); fi
  %-scaling---------------------------------------------------------------------------------------
  if blength>1: blen:=blength;        proc_size_setup; proc_skeleton; proc_scaling;
  ef blength>0: blen:=fig_wd*blength; proc_size_setup; proc_skeleton; proc_scaling;
  else:
    blen:=3mm;
    proc_size_setup;
    if xpart(msize)<1: m_wd:=fig_wd*xpart(msize); else: m_wd:=fig_wd; fi
    if ypart(msize)<1: m_ht:=fig_ht*ypart(msize); else: m_ht:=fig_ht; fi
    for i=1 upto 6:
      proc_skeleton; proc_scaling;
      if (mol_ht/mol_wd)>(m_ht/m_wd):
        if ypart(msize)>1: factor:=ypart(msize)/mol_ht;
        else: factor:=((fig_ht-2ypart(fmargin))*ypart(msize))/mol_ht; fi
      else: if xpart(msize)>1: factor:=xpart(msize)/mol_wd;
            else: factor:=((fig_wd-2xpart(fmargin))*xpart(msize))/mol_wd; fi fi
            exitif (factor>=1-eps)and(factor<=1+eps); blen:=blen*factor; proc_size_setup;
    endfor
    if blen>max_blength: blen:=max_blength; proc_size_setup; proc_skeleton; proc_scaling; fi
  fi
  %-----------------------------------------------------------------------------------------------
  for i=1 upto cntA:
    if addA[i]<>0:
      if addA[i]=PLS: chargeA[i]:=1; ef addA[i]=MIS: chargeA[i]:=-1; else: chargeA[i]:=0; fi
    else: chargeA[i]:=0; fi
  endfor
  %===============================================================================================
  if scan_bit(sw_output,Fig):
    %-draw atom-----------------------------------------------------------------------------------
    if sw_numbering=0: for i=1 upto cntA: if strA[i]<>0: draw_atom(i); fi endfor fi
    %-draw add to atom----------------------------------------------------------------------------
    if (not scan_bit(sw_numbering,Atom))and(not scan_bit(sw_numbering,Bond)):
      for i=1 upto cntA:
        if addA[i]<>0: draw_char(char(addA[i]),sfrt(posA[i],atom_wd,angX[i]+add_rot[i])); fi
      endfor fi
    %-draw bond-----------------------------------------------------------------------------------
    for i=1 upto cntB: if lineB[i]<si_ : draw_bond(i); fi endfor
    for i=1 upto cntB: if lineB[i]>=si_: draw_bond(i); fi endfor
    %-atom numbering------------------------------------------------------------------------------
    if scan_bit(sw_numbering,Atom):
      if scan_bit(sw_numbering,All): numbering_end:=cntA; else: numbering_end:=backboneA; fi
      for i=1 upto numbering_end:
        defaultscale:=.2blen/defaultsize; nH:=1.2defaultsize*defaultscale;
        if i<=9: nW:=nH; ef i<=99: nW:=1.3nH; else: nW:=1.9nH; fi
        erase fill fullcircle xscaled 1.1nW yscaled 1.1nH shifted posA[i] wpcs 0.1;
        draw fullcircle xscaled 1.1nW yscaled 1.1nH shifted posA[i] wpcs 0.1;
        if numberfont="draw": draw_number(i,posA[i]); else: label(decimal(i),posA[i]); fi
      endfor
    fi
    %-bond numbering------------------------------------------------------------------------------
    if scan_bit(sw_numbering,Bond):
      if scan_bit(sw_numbering,All): numbering_end:=cntB; else: numbering_end:=backboneB; fi
      for i=1 upto numbering_end:
        defaultscale:=.18blen/defaultsize; nH:=1.2defaultsize*defaultscale;
        if i<=9: nW:=nH; ef i<=99: nW:=1.3nH; else: nW:=1.9nH; fi
        nH:=defaultsize*defaultscale; tpos:=.5[posA[sB[i]],posA[eB[i]]];
        erase fill unitsquare xscaled nW yscaled 1.05nH shifted (tpos-(nW/2,1.05nH/2));
        draw unitsquare xscaled nW yscaled 1.05nH shifted (tpos-(nW/2,1.05nH/2)) wpcs 0.1;
        if numberfont="draw": draw_number(i,tpos); else: label(decimal(i),tpos); fi
      endfor
    fi
    %-make picture -------------------------------------------------------------------------------
    if xpart(mposition)>1: nX:=xpart(mposition)-minX;
    else: nX:=xpart(fmargin)-minX+(fig_wd-mol_wd-2xpart(fmargin))*xpart(mposition); fi
    if ypart(mposition)>1: nY:=ypart(mposition)-minY;
    else: nY:=ypart(fmargin)-minY+(fig_ht-mol_ht-2ypart(fmargin))*ypart(mposition); fi
    posM[0][incr cntM]:=(nX,nY); posM1[cntM]:=(minX+nX,minY+nY); posM2[cntM]:=(maxX+nX,maxY+nY);
    wdM[cntM]:=mol_wd; htM[cntM]:=mol_ht;
    mol_pic[cntM]:=currentpicture;
    clearit;
  fi
  %-Calculate information ------------------------------------------------------------------------
  MW_n:=MI_n:=tbl_atom_max:=warning_cnt:=hideH_cnt:=0; nA:=pc_emb_start;
  for i=1 upto tbl_atom_end: sumA[i]:=0; endfor
  for i=1 upto cntA:
    knownA:=bond_cntA:=0; nS:=strA[i];
    for j=1 upto cntB: bond_num[j]:=bond_type(lineB[j]);
                       if (sB[j]=i)or(eB[j]=i): bond_cntA:=bond_cntA+bond_num[j]; fi
    endfor
    Bcnt[i]:=bond_cntA;
    if ((nS=0)or(nS=C-nA))and(bond_cntA<4): hideH[i]:=4-bond_cntA; hideH_cnt:=hideH_cnt+hideH[i];
    else: hideH[i]:=0; fi
    if (nS=0)or(nS=C-nA): if bond_cntA>4:  warn_valence(i) fi
    ef nS=(O-nA): if bond_cntA<>2: warn_valence(i) fi
    ef nS=(N-nA): if (bond_cntA<>3)and(bond_cntA<>5): warn_valence(i) fi
    ef nS=(S-nA): if (bond_cntA<>2)and(bond_cntA<>4)and(bond_cntA<>6): warn_valence(i) fi
    ef nS=(P-nA): if bond_cntA<>5: warn_valence(i) fi
    ef (nS=H-nA)or(nS=OH-nA)or(nS=COOH-nA)or(nS=NH2-nA)or(nS=CN-nA)or(nS=F-nA)or
       (nS=Cl-nA)or(nS=Br-nA): if bond_cntA<>1: warn_valence(i) fi fi
    for j=1 upto tbl_grp_end:
      if str_tbl[nS]=tbl_atoms[j]:
        if tbl_atom[j]=0: sumA[j]:=sumA[j]+1; if j>tbl_atom_max: tbl_atom_max:=j; fi
        else: for k=1 upto tbl_atom[j]:
                sumA[tbl_grp[j][k]]:=sumA[tbl_grp[j][k]]+1;
                if tbl_grp[j][k]>tbl_atom_max: tbl_atom_max:=tbl_grp[j][k]; fi
              endfor fi
        knownA:=1; fi
      exitif knownA=1;
    endfor
    if knownA=0: message " Unknown Str("&str_tbl[strA[i]]&") is used "&decimal(i); fi
  endfor
  sumA[2]:=sumA[2]+hideH_cnt; if (tbl_atom_max=1)and(sumA[2]>0): tbl_atom_max:=2; fi
  for i=1 upto tbl_atom_max:
    if sumA[i]>=1: MW_n:= MW_n+tbl_atom_wt[i]*sumA[i]; MI_n:= MI_n+tbl_atom_mi[i]*sumA[i];
                   fm:=fm&erase_char(tbl_atoms[i]) if sumA[i]>=2: &decimal(sumA[i]) fi; fi
  endfor
  mw:=substring(0,8) of decimal(MW_n); mi:=substring(0,8) of decimal(MI_n);
  endgroup; % [;]
enddef;
%-------------------------------------------------------------------------------------------------
def add_grp(expr p)=
  if f_at=1: nA:=adrT; else: nA:=cntA+1; fi
  if checkA(nA):
    grp_adr[incr grp_num]:=nA;
    if lenL<>lenC: if lenL>=0: grp_len[grp_num]:=grp_lenL[grp_num]:=-lenL;
                   else:       grp_len[grp_num]:=grp_lenL[grp_num]:=lenL; fi
    else: grp_len[grp_num]:=lenT; grp_lenL[grp_num]:=lenC; fi
    grp_line[grp_num]:=lineT; grp_env[grp_num]:=envT;
    grp_rot[grp_num]:=rotT; grp_parts[grp_num]:=p;
    if f_lineT=0: lineT:=si; fi
    if f_lenT=0: lenT:=lenC; fi
    if f_rotT=0: rotT:=0; fi
    if f_envT=0: envT:=hz; fi
  fi
enddef;
%-------------------------------------------------------------------------------------------------
def expand_grp(expr n)=
  save_grp_num:=grp_num; save_mc_cnt:=mc_cnt;
  for i=n upto grp_num:
    put_grp(_jp_absA,grp_adr[i]) put_grp(_com,_mark)
    if grp_line[i]<>si: 
      if grp_line[i]=nl: put_grp(_tmp_len,_size_atom) put_grp(_adj_ang,0) fi
      put_grp(_tmp_line,grp_line[i]) fi
    if (grp_len[i]<>lenC)and(grp_line[i]<>nl): put_grp(_tmp_len,grp_len[i]) fi
    if grp_env[i]<>hz: put_grp(_chg_env,grp_env[i]) fi
    put_grp(_mk_bond,grp_rot[i])
    if grp_parts[i]<>NO_ATOM:
      if grp_lenL[i]<>lenC: put_grp(_chg_len,grp_lenL[i]) fi
      for j=1 upto pc_cnt[grp_parts[i]]:
        if pc_x[grp_parts[i]][j]>=pc_emb_start: expand_pcode(pc_x[grp_parts[i]][j],0)
        else: put_grp(pc_x[grp_parts[i]][j],pc_y[grp_parts[i]][j]) fi
      endfor
    fi
    put_grp(_chg_env,hz) put_grp(_com,_term) put_grp(_com,_rest) put_grp(_chg_len,_end)
  endfor
  proc_bond_atom(save_mc_cnt+1);
  if grp_num>save_grp_num: expand_grp(save_grp_num+1); fi
enddef;
%-------------------------------------------------------------------------------------------------
def put_grp(expr x,y)=
  mc_x[incr mc_cnt]:=x; mc_y[mc_cnt]:=y; mc_b[mc_cnt]:=0; enddef;
%=================================================================================================
def draw_frame(expr o,p,n)=
  draw ((0,0)--(xpart p,0)--p--(0,ypart p)--cycle) shifted o withpen pensquare scaled n; enddef;
def draw_corner(expr p,n)=
  draw (0,0) wpcs n; draw(xpart p,0) wpcs n; draw p wpcs n; draw(0,ypart p) wpcs n; enddef;
%-------------------------------------------------------------------------------------------------
def proc_size_setup=
  atom_wd:=     blen*ratio_atom_bond+offset_atom;
  wedge_wd:=    blen*ratio_wedge_bond+offset_wedge;
  hash_gap:=    blen*ratio_hashgap_bond+offset_hash_gap;
  bondgap:=     blen*ratio_bondgap_bond+offset_bond_gap;
  bond_pen_wd:= blen*ratio_thickness_bond+offset_thickness;
enddef;
%-------------------------------------------------------------------------------------------------
def proc_scaling=
  minX:=minY:=4095; maxX:=maxY:=-4095;
  for i=1 upto cntA:
    nX:=xpart(posA[i]); nY:=ypart(posA[i]);
    if strA[i]<>0:
      nU:=nD:=nP:=nL:=nR:=0;
      for j=1 upto length(str_tbl[strA[i]]):
        tempc:=substring(j-1,j) of str_tbl[strA[i]];
        if tempc="^": nU:=.5atom_wd; ef tempc="_": nD:=.5atom_wd;
        ef (tempc<>"{")and(tempc<>"}"): nP:=nP+atom_wd*tbl_char_wd[ASCII(tempc)]; fi
      endfor
      if (angX[i]<=90)or(angX[i]>=270): nR:=nP; else: nL:=nP; fi
      if (nX-nL+.5atom_wd)<minX: minX:=nX-nL+.5atom_wd; fi
      if (nX+nR-.5atom_wd)>maxX: maxX:=nX+nR-.5atom_wd; fi
      if (nY-nD-.5atom_wd)<minY: minY:=nY-nD-.5atom_wd; fi
      if (nY+nU+.5atom_wd)>maxY: maxY:=nY+nU+.5atom_wd; fi
    else: if nX<minX: minX:=nX; fi if nX>maxX: maxX:=nX; fi
          if nY<minY: minY:=nY; fi if nY>maxY: maxY:=nY; fi fi
  endfor
  mol_wd:=maxX-minX; mol_ht:=maxY-minY;
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def proc_bond_atom(expr n)=
  f_bra:=f_term:=rotT:=f_lineT:=f_rotT:=f_lenT:=f_envT:=temp_strA:=f_at:=f_inv:=addAT:=markA:=0;
  markB:=saveA:=saveB:=sB[0]:=0; eB[0]:=1; lenL:=lenT:=lenC; lineT:=si; envB:=envT:=hz;
  %-----------------------------------------------------------------------------------------------
  for i=n upto mc_cnt:
    pX:=mc_x[i]; pY:=mc_y[i]; pB:=mc_b[i];
    if pX=_mk_bond: if (pY=0)and(rotT<>0): rotT:=0; fi add_atom(0);
    ef pX=_set_adrA: if (pY mod 1)>0: f_inv:=1; pY:=round pY; fi adrT:=getA(pY);
    ef pX=_set_adrB: if (pY mod 1)>0: f_inv:=1; pY:=round pY; fi adrT:=getB(pY);
    ef pX=_com: if pY=_mark: saveA:=markA; saveB:=markB; markA:=cntA; markB:=cntB;
                ef pY=_rest: markA:=saveA; markB:=saveB;
                ef pY=_moff: markA:=markB:=0;
                ef pY=_term: termA;
                ef pY=_grp_s: f_at:=1; if lineT<>si: f_lineT:=1; fi if rotT<>0: f_rotT:=1; fi
                                if lenT<>lenC: f_lenT:=1; fi if envT<>hz: f_envT:=1; fi
                ef pY=_grp_e: f_at:=f_lineT:=f_rotT:=f_lenT:=f_envT:=rotT:=0;
                                lineT:=si; lenT:=lenC; envT:=hz; fi
    ef pX=_set_atom: temp_strA:=pY;
    ef pX=_grp_si: add_grp(pY);
    ef pX=_grp_dm: lineT:=dm; add_grp(pY);
    ef pX=_grp_wf: if f_inv=1: lineT:=zf; f_inv:=0; else: lineT:=wf; fi add_grp(pY);
    ef pX=_grp_zf: if f_inv=1: lineT:=wf; f_inv:=0; else: lineT:=zf; fi add_grp(pY);
    ef pX=_jp_bond: nB:=getB(pY); nB:=checkBn(nB); termA; sB[cntB+1]:=sB[nB]; f_bra:=1;
    ef pX=_jp_atom: nA:=getA(pY); nA:=checkAn(nA); termA; sB[cntB+1]:=nA; f_bra:=1;
    ef pX=_jp_absA: sB[cntB+1]:=pY; f_bra:=1; temp_cntB:=cntB;
    ef pX=_chg_atom: if checkA(adrT): strA[adrT]:=pc_y[pY][1]; fi
    ef pX=_tmp_len: lenT:=pY;
    ef pX=_chg_len: if pY=_end: lenL:=save_lenS; else: save_lenS:=lenL; lenL:=pY; fi
    ef pX=_set_len: if pY=_end: lenL:=save_lenL;
                    ef pY=_ring: save_lenL:=lenL; if lenT<>lenC:lenL:=lenT; ef lenL<0:lenL:=1; fi
                    ef pY=_chain: save_lenL:=lenL; if lenT<>lenC:lenL:=lenT; fi
                    else: save_lenL:=lenL; lenL:=pY; fi
    ef pX=_get_len:  nB:=getB(pY); save_lenL:=lenL; lenL:=lenB[nB];
    ef pX=_set_line: if checkB(adrT):
                       if f_inv=1: f_inv:=0; if pY=dl: pY:=dr; ef pY=dr: pY:=dl; fi fi
                       lineB[adrT]:=pY; fi
    ef pX=_tmp_line: lineT:=pY;
    ef pX=_tmp_rot:  rotT:=pY;
    ef pX=_cyc:      nA:=getA(pY); nA:=checkAn(nA); add_atom(nA);
    ef pX=_cyc_eB:   nB:=getB(pY); nB:=checkBn(nB); add_atom(eB[nB]);
    ef pX=_cyc_sB:   nB:=getB(pY); nB:=checkBn(nB); add_atom(sB[nB]);
    ef pX=_chg_env:  envB:=pY;
    ef pX=_tmp_env:  envT:=pY;
    ef pX=_set_colorA: colorA[adrT]:=pY;
    ef pX=_set_colorB: colorB[adrT]:=pY;
    ef pX=_add_charge: addAT:=pY;
    ef pX=_set_charge: if checkA(adrT): addA[adrT]:=pY; if rotT<>0: add_rot[adrT]:=rotT; fi fi
    fi
  endfor
enddef;
%-------------------------------------------------------------------------------------------------
def add_atom(expr n)=
  lineB[incr cntB]:=lineT; lineT:=si;
  if lenT=lenC: lenB[cntB]:=lenL; else: lenB[cntB]:=lenT; lenT:=lenC; fi
  if f_bra=0: strA[incr cntA]:=temp_strA; sB[cntB]:=cntA;
              addA[cntA]:=addAT; addAT:=temp_strA:=add_rot[cntA]:=0;
              if rotT<>0: add_rot[cntA]:=rotT; rotT:=0; fi
  else: f_bra:=0; fi
  if n=0: eB[cntB]:=cntA+1; f_term:=0; else: eB[cntB]:=n; f_term:=1; fi
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def proc_skeleton=
  markA:=markB:=cntA:=cntB:=f_bra:=rotT:=f_term:=angA[0]:=angB[0]:=angX[0]:=0;
  lineT:=si; envT:=envB:=hz; angT:=mangle; posA[0]:=posBs:=posBe:=(0,0);
  %-----------------------------------------------------------------------------------------------
  for i=1 upto mc_cnt: pX:=mc_x[i]; pY:=mc_y[i]; pB:=mc_b[i];
    if pX=_mk_bond: if (pY=0)and(rotT<>0):pY:=rotT; rotT:=0; fi add_bond(pY,1);
    ef pX=_com: if pY=_mark: saveA:=markA; saveB:=markB; markA:=cntA; markB:=cntB;
                ef pY=_rest: markA:=saveA; markB:=saveB;
                ef pY=_moff: markA:=markB:=0; ef pY=_term: termB; fi
    ef pX=_jp_bond: adrT:=getB(pY); adrT:=checkBn(adrT);
                    termB; posBs:=posA[sB[adrT]]; angT:=angB[adrT]; f_bra:=1; rotT:=0;
    ef pX=_jp_atom: adrT:=getA(pY); adrT:=checkAn(adrT); 
                    termB; posBs:=posA[adrT]; angT:=angX[adrT]; f_bra:=1; rotT:=0;
    ef pX=_jp_absA: adrT:=pY; posBs:=posA[adrT]; angT:=angX[adrT];
                    f_bra:=1; rotT:=0; temp_cntB:=cntB;
    ef pX=_adj_ang: angT:=adjust_ang(angT);
    ef pX=_rot_ang: if pY>-3700: angT:=(angT+pY) mod 360; else: angT:=(pY+4095) mod 360; fi
    ef pX=_tmp_rot: rotT:=pY;
    ef pX=_chg_env: envB:=pY;
    ef pX=_tmp_env: envT:=pY;
    ef pX=_cyc:     nA:=getA(pY); nA:=checkAn(nA); add_bond(angle(posA[nA]-posBs)-angT,0);
    ef pX=_cyc_sB:  nA:=getB(pY); nA:=checkBn(nA); add_bond(angle(posA[sB[nA]]-posBs)-angT,0);
    ef pX=_cyc_eB:  nA:=getB(pY); nA:=checkBn(nA); add_bond(angle(posA[eB[nA]]-posBs)-angT,0);
    fi
  endfor
enddef;
%-------------------------------------------------------------------------------------------------
def add_bond(expr n,f)=
  if n=_auto_ang: nA:=arrange_ang(angT mod 360); else: nA:=n; fi
  if f_bra=0:
    adrT:=incr cntA; posA[cntA]:=posBs; angA[cntA]:=angT; 
    angX[cntA]:=(angT+nA/2+iif(nA>=0,-90,90)) mod 360;
  else: f_bra:=0; fi
  cntB:=cntB+1;
  if nA>-3700: angB[cntB]:=angT:=(angT+nA) mod 360; else: angB[cntB]:=angT:=nA+4095; fi
  if f=1:
    if lenB[cntB]=_size_atom: posBe:=sfrt(posBs,1.1atom_wd,angT);
    else: nA:=lenB[cntB]; if nA<0:nB:=glu_atom(adrT)+glu_atom(cntA+1); nA:=abs(nA); else:nB:=0; fi
          posBe:=sfrt(posBs,nA*blen+nB,angT); fi
    posA[cntA+1]:=posBe; f_term:=0;
  else: f_term:=1; fi
  posBs:=posBe;
enddef;
%-------------------------------------------------------------------------------------------------
vardef arrange_ang(expr n)=
  if f_bra=1: 0
  else: if cntB=0: angT:=(angT-180) mod 360; 180
        else: if envB>=pc_emb_start: pc_x[envB][cntB-temp_cntB]
              else: if envB=hz: if n=0:60 ef n<=90:-60 ef n<=180:60 ef n<270:-60 else:60 fi
                    ef envB=vt: if n=0:-60 ef n<90:60 ef n<=180:-60 ef n<=270:60 else:-60 fi
                    ef envB=lr: if odd(cntB-temp_cntB):60 else:-60 fi
                    ef envB=rl: if odd(cntB-temp_cntB):-60 else:60 fi
                    ef abs(envB)<=180: envB  fi fi fi
  fi
enddef;
%-------------------------------------------------------------------------------------------------
vardef adjust_ang(expr n)= if (n<40)or(n>320): 0 ef n<140: 90 ef n<220: 180 else: 270 fi enddef;
%=================================================================================================
vardef getA(expr n)= if n>=0: markA+n ef n>=-999: cntA+n+1 else: n+4095 fi enddef;
vardef getB(expr n)= if n>=0: markB+n ef n>=-999: cntB+n+1 else: n+4095 fi enddef;
vardef checkA(expr n)=
  bA:=(n<=(cntA+iif(f_term=0,1,0)))and(n>=1)and(n mod 1=0); if not bA: message_error(pB,4) fi bA
enddef;
vardef checkB(expr n)= bA:=(n<=cntB)and(n>=1)and(n mod 1=0); if not bA: message_error(pB,5) fi bA
enddef;
vardef checkAn(expr n)= if checkA(n): n else: 1 fi enddef;
vardef checkBn(expr n)= if checkB(n): n else: 1 fi enddef;
%-------------------------------------------------------------------------------------------------
def termA=
  if f_term=0: if f_bra=0: strA[incr cntA]:=temp_strA; addA[cntA]:=addAT; add_rot[cntA]:=rotT;
                           addAT:=temp_strA:=rotT:=0;
               else: f_bra:=0; fi
               f_term:=1; fi
enddef;
def termB= if f_term=0: if f_bra=0: angX[incr cntA]:=angT; else:f_bra:=0; fi f_term:=1; fi enddef;
%-------------------------------------------------------------------------------------------------
vardef glu_atom(expr n)=
  if strA[n]<>0: nD:=angT mod 90; nC:=0.5atom_wd; nN:=(iif(nD<45,sind nD,cosd nD)*nC)PADD nC; nN
  else: 0 fi
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def draw_atom(expr n)=
  if atomfont<>"draw": defaultfont:=atomfont; defaultsize:=atom_wd; fi
  atoms:=str_tbl[strA[n]]; slen:=length(atoms); nC:=nS:=0; raise_pos:=(0,0); tpos:=posA[n];
  if (angX[n]<=90)or(angX[n]>=270): sdir:=1; else: sdir:=-1; fi
  for i=1 upto slen:
    if nC=0: tempc:=subc(atoms,i);
      if (sdir=-1)and(tempc="{"):
        nD:=i; nC:=0; for j=nD upto slen: nC:=nC+1; exitif subc(atoms,j+1)="}"; endfor
      fi
    else: nC:=nC-1; tempc:=subc(atoms,nD+nC);
    fi
    if tempc="_": raise_pos:=iif(raise_pos=(0,0),(0,-.5atom_wd),(0,0));
    ef tempc="^": raise_pos:=iif(raise_pos=(0,0),(0, .5atom_wd),(0,0));
    ef (tempc<>"{")and(tempc<>"}"):
      nS:=nS+1; char_wd:=atom_wd*tbl_char_wd[ASCII(tempc)]; char_ht:=atom_wd;
      if nS=1: if (sdir=-1)and(char_wd<atom_wd): tpos:=tpos+((atom_wd-char_wd)/2,0); fi 
      else: tpos:=tpos+(.5char_wd*sdir,0); fi
      tcol:=colorA[n]; f_col:=0;
      if known tcol: if tcol<>0: drawoptions(withcolor color_list[tcol]); f_col:=1; fi fi
      if atomfont="draw": draw_char(tempc,tpos+raise_pos); else: label(tempc,tpos+raise_pos); fi
      if f_col=1: drawoptions(); fi
      tpos:=tpos+(.5char_wd*sdir,0);
    fi
  endfor
  nA:=0.56atom_wd; nB:=0.06atom_wd;
  if sdir=1: frameA[n]:=posA[n]-(nA,nA)--tpos+(nB,-nA)--tpos+(nB,nA)--posA[n]+(-nA,nA)--cycle;
  else:      frameA[n]:=tpos-(nB,nA)--posA[n]+(nA,-nA)--posA[n]+(nA,nA)--tpos+(-nB,nA)--cycle; fi
  if scan_bit(sw_frame,Atom): draw frameA[n] wpcs thickness_frame; fi
enddef;
%-------------------------------------------------------------------------------------------------
def draw_number(expr n,p)=
  atom_wd:=.18blen; bond_pen_wd:=.012blen; temps:=decimal(n); slen:=length(temps); spos:=p;
  for i=1 upto slen:
    if i=1: spos:=spos-(slen*.35atom_wd-.35atom_wd,0); else: spos:=spos+(0.73atom_wd,0); fi 
    draw_char(substring(i-1,i) of temps,spos);
  endfor
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def draw_bond(expr n)=
  nL:=lineB[n]; angL:=angB[n]; nS:=sB[n]; nE:=eB[n]; f_col:=0;
  tcol:=colorB[n]; zL:=posA[nS]--posA[nE]; ww:=wedge_wd; ap:=angL+90; am:=angL-90; aw:=atom_wd;
  %-----------------------------------------------------------------------------------------------
  if (strA[nS]=0)and(strA[nE]=0)or(sw_numbering>=1): Ls:=posA[nS]; Le:=posA[nE]; pA:=(.1,.9);
  ef strA[nS]=0: Ls:=posA[nS]; Le:=zL intersectionpoint frameA[nE]; pA:=(.15,1);
  ef strA[nE]=0: Ls:=zL intersectionpoint frameA[nS]; Le:=posA[nE]; pA:=(0,.85);
  else: Ls:=zL intersectionpoint frameA[nS]; Le:=zL intersectionpoint frameA[nE]; pA:=(0,1); fi
  zA:=Ls--Le; lenL:=length(Le-Ls);
  %-----------------------------------------------------------------------------------------------
  if known tcol: if tcol<>0: drawoptions(withcolor color_list[tcol]); f_col:=1;fi fi
  pickup pencircle scaled bond_pen_wd;
  if (nL=si)or(scan_bit(sw_omit,Bond)): draw zA;
  ef nL=dl: draw zA; draw sfrt(subpath pA of zA,bondgap,ap);
  ef nL=dr: draw zA; draw sfrt(subpath pA of zA,bondgap,am);
  ef nL=dm: draw sfrt(zA,bondgap/1.75,ap); draw sfrt(zA,bondgap/1.75,am);
  ef nL=db: nA:=iif(((angL-angX[nS]) mod 360)<=180,ap,am);
            draw zA; draw sfrt(subpath pA of zA,bondgap,nA);
  ef nL=wf: fill Ls--sfrt(Le,ww,am)--sfrt(Le,ww,ap)--cycle;
  ef nL=zf: wz_put(Ls,sfrt(Le,ww,ap),sfrt(Le,ww,am));
  ef nL=nl:
  ef nL=wb: fill sfrt(Ls,ww,am)--Le--sfrt(Ls,ww,ap)--cycle;
  ef nL=tm: draw zA; draw sfrt(zA,bondgap,ap); draw sfrt(zA,bondgap,am);
  ef nL=bd: draw zA withpen penrazor rotated ap scaled bondgap;
  ef nL=bz: bz_put(sfrt(Ls,ww,ap),sfrt(Le,ww,ap),sfrt(Ls,ww,am),sfrt(Le,ww,am));
  ef nL=zb: wz_put(Le,sfrt(Ls,ww,am),sfrt(Ls,ww,ap));
  ef nL=dt: for i=0 step .75hash_gap/lenL until 1: drawdot i[Ls,Le] wpcs 2bond_pen_wd; endfor
  ef nL=wv: nA:=.4bondgap; nB:=round(lenL/nA);
            draw Ls for i=1 upto nB: ..controls(point (i-.5)/nB of sfrt(zA,nA,iif(odd i,ap,am)))
                                     ..point i/nB of zA endfor
  ef nL=wf_r: filldraw Ls--sfrt(Le,.35ww,am)--sfrt(Le,.35ww,ap)--cycle wpcs .05ww;
  ef nL=wb_r: filldraw sfrt(Ls,.35ww,am)--Le--sfrt(Ls,.35ww,ap)--cycle wpcs .05ww;
  ef nL=bd_r: draw zA wpcs .65bondgap;
  ef nL=vf:  draw zA;draw sfrt(Le,bondgap,angL-150)--Le--sfrt(Le,bondgap,angL+150);
  ef nL=vb:  draw zA;draw sfrt(Ls,bondgap,angL-30)--Ls--sfrt(Ls,bondgap,angL+30);
  ef nL=si_: erase draw subpath (.15,.85) of zA wpcs 0.8bondgap; draw zA;
  ef nL=dl_: erase draw subpath (.15,.85) of sfrt(subpath pA of zA,.5bondgap,ap) wpcs 1.8bondgap;
             draw zA; draw sfrt(subpath pA of zA,bondgap,ap);
  ef nL=dr_: erase draw subpath (.15,.85) of sfrt(subpath pA of zA,.5bondgap,am) wpcs 1.8bondgap;
             draw zA; draw sfrt(subpath pA of zA,bondgap,am);
  ef nL=dm_: erase draw subpath(0.15,0.85) of zA wpcs 1.8 bondgap;
             draw sfrt(zA,bondgap/1.75,ap); draw sfrt(zA,bondgap/1.75,am);
  ef nL=wf_: erase draw subpath (0.15,0.85) of (Ls--sfrt(Le,ww,am)) wpcs 0.8bondgap;
             erase draw subpath (0.15,0.85) of (Ls--sfrt(Le,ww,ap)) wpcs 0.8bondgap;
             fill Ls--sfrt(Le,ww,am)--sfrt(Le,ww,ap)--cycle;
  ef nL=wb_: erase draw subpath (0.15,0.85) of (sfrt(Ls,ww,am)--Le) wpcs 0.8bondgap;
             erase draw subpath (0.15,0.85) of (sfrt(Ls,ww,ap)--Le) wpcs 0.8bondgap;
             fill sfrt(Ls,ww,am)--Le--sfrt(Ls,ww,ap)--cycle;
  ef nL=zf_: erase draw subpath (0.15,0.85) of (Ls--sfrt(Le,ww,am)) wpcs 0.8bondgap;
             erase draw subpath (0.15,0.85) of zA wpcs 0.8bondgap;
             erase draw subpath (0.15,0.85) of (Ls--sfrt(Le,ww,ap)) wpcs 0.8bondgap;
             wz_put(Ls,sfrt(Le,ww,ap),sfrt(Le,ww,am));
  ef nL=zb_: erase draw subpath (0.15,0.85) of (sfrt(Ls,ww,am)--Le) wpcs 0.8bondgap;
             erase draw subpath (0.15,0.85) of zA wpcs 0.8bondgap;
             erase draw subpath (0.15,0.85) of (sfrt(Ls,ww,ap)--Le) wpcs 0.8bondgap;
             wz_put(Le,sfrt(Ls,ww,am),sfrt(Ls,ww,ap));
  ef nL=bd_: erase draw subpath(0.15,0.85) of zA wpcs 1.6bondgap;
             draw zA withpen penrazor rotated ap scaled bondgap;
  %-- bond type for glycan ----------------------------------------------------------------------
  ef nL=arc_lb:  draw Ls--Ls-(0,aw)..posA[nE]+(-1.2aw,0)..posA[nE]-(.6aw,0);
  ef nL=arc_br:  draw posA[nS]+(.6aw,0)..posA[nS]+(1.2aw,0)..Le-(0,aw)--Le;
  ef nL=arc_lbr: draw posA[nS]+(0,iif(strA[nS]=0,0,-.6aw))--posA[nS]+(0,-.8aw)
                      ..0.5[posA[nS],posA[nE]]+(0,-1.7aw)..posA[nE]+(0,-.8aw)
                      --posA[nE]+(0,iif(strA[nE]=0,0,-.6aw));
  ef nL=arc_ltr: draw posA[nS]+(0,iif(strA[nS]=0,0,0.6aw))--posA[nS]+(0,.8aw)
                      ..0.5[posA[nS],posA[nE]]+(0,1.7aw)..posA[nE]+(0,.8aw)
                      --posA[nE]+(0,iif(strA[nE]=0,0,.6aw)); fi
  if f_col=1: drawoptions(); fi
enddef;
%------------------------------------------------------------------------------------------------
def wz_put(expr a,b,c)=
  nB:=round(lenL/hash_gap);
  for i=1 upto nB: nA:=i/nB; if i=1: nD:=0; else: nD:=(i-ratio_hash_black)/nB; fi
                   fill nD[a,b]--nD[a,c]--nA[a,c]--nA[a,b]--cycle;
  endfor
enddef;
def bz_put(expr a,b,c,d)=
  nB:=round(lenL/hash_gap);
  for i=0 upto nB-1: nA:=i/nB; nD:=nA+ratio_hash_black/nB;
                     fill nA[b,a]--nA[d,c]--nD[d,c]--nD[b,a]--cycle;
  endfor
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def char_size_set(expr w,h)(expr s)=
  for j=1 upto length(s): nA:=ASCII(subc(s,j)); tbl_char_wd[nA]:=w; tbl_char_ht[nA]:=h; endfor
enddef;
%-------------------------------------------------------------------------------------------------
char_size_set(   1,   1)("CGHMNOQW");
char_size_set( 0.9,   1)("ABDFIJKPRSTUVXY");
char_size_set( 0.8,   1)("ELZ");
char_size_set( 0.7,   1)("0123456789nhtfg");
char_size_set( 0.7, 0.7)("-+");
char_size_set(0.45,0.95)("l");
char_size_set(0.75, 0.8)("opq");
char_size_set( 0.8, 0.8)("e");
char_size_set( 0.9, 0.8)("wm");
char_size_set( 0.7, 0.8)("abdcksuvrxyz");
char_size_set(0.35, 0.9)("i");
char_size_set( 0.5, 0.9)("j");
%-------------------------------------------------------------------------------------------------
def dw expr p = draw p shifted cpos enddef;
def dwv expr p = draw p withpen penrazor scaled fP shifted cpos enddef;
def dwvs(expr n) expr p = draw p withpen penrazor scaled (fP*n) shifted cpos enddef;
def dwh expr p = draw p withpen penrazor rotated 90 scaled fP shifted cpos enddef;
def cdw expr p = cutdraw p shifted cpos enddef;
%-------------------------------------------------------------------------------------------------
def Z_a=( 0,hP) enddef; def Z_b=(hP, 0) enddef; def Z_c=(hP,hP) enddef; def Z_d=(aW,hP) enddef;
def Z_e=(fW, 0) enddef; def Z_f=(hW,aH) enddef; def Z_g=(hW, 0) enddef; def Z_h=( 0,hH) enddef;
def Z_i=(hW,fW) enddef; def Z_j=( 0,qH) enddef; def Z_k=(aW,qH) enddef; def Z_l=(.75aW,0) enddef;
def Z_m=(hP,hH) enddef; def Z_n=(fW,fH) enddef; def Z_o=(fW,hH) enddef; def Z_p=(hW,aW) enddef;
def Z_q=( 0,fH) enddef; def Z_r=(hP,fH) enddef; def Z_s=(hW,fH) enddef; def Z_t=(fW,aH) enddef;
def Z_u=(aW,fH) enddef; def Z_v=(aW,hH) enddef; def Z_w=(hP,aH) enddef; def Z_x=(hW,hP) enddef;
def Z_y=(hW,hH) enddef; def Z_z=(fW,hP) enddef; 
%-------------------------------------------------------------------------------------------------
def circ_O = Z_o..(.8aW,fH-qP)..tension 1.5..(.2aW,fH-qP)..Z_m..
     (.2aW,1.5hP)..tension 1.5..(.8aW,1.5hP)..cycle enddef;
def circ_Oh = (hP,qH)..Z_x..(fW,qH)..Z_y..cycle enddef;
def circ_sOh = (hP,qH)..Z_x..(.98fW,qH)..Z_y..cycle enddef;
def circ_Oa = (hP,0.35aH)..Z_x..(fW,0.35aH)..(hW,.7aH)..cycle enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def draw_char(expr s,p)=
  aW:=atom_wd*tbl_char_wd[ASCII(s)]*(1-2ratio_atomgap_atom);
  aH:=atom_wd*tbl_char_ht[ASCII(s)]*(1-2ratio_atomgap_atom);
  cpos:=p-(aW/2,atom_wd/2*(1-2ratio_atomgap_atom));
  fP:=atom_wd*ratio_char_atom;
  hP:=fP/2; qP:=fP/4; fW:=aW-hP; hW:=aW/2; fH:=aH-hP; hH:=aH/2; qH:=aH/4;
  %-----------------------------------------------------------------------------------------------
  pickup pencircle scaled fP;
  if s="C": cdw sbp(.05,.95)circ_O;
  ef s="H": dwv Z_b--Z_w; dw Z_m--Z_o; dwv Z_e--Z_t;
  ef s="O": dw circ_O;
  ef s="N": dwv Z_b--Z_w; dwv Z_e--Z_t; dwvs(1.4)(1.4hP,aH)--(aW-1.4hP,0);
  ef s="S": cdw sbp(.05,.45)circ_O; cdw sbp(.55,.95)circ_O; dw (fW,.3aH){up}..{up}(hP,.7aH);
  ef s="F": dwh Z_q--Z_u; dwh (.1hP,.48aH)--(fW-hP,.48aH); dwv Z_b--Z_r;
  ef s="P": dwv Z_b--Z_w; dw Z_r--(.65aW,fH){right}..(fW,.7aH)..{left}(.65aW,.44aH)..(hP,.44aH);
  ef s="I": dwv Z_x--Z_s; dwh (hW-fP,hP)--(hW+fP,hP); dwh (hW-fP,fH)--(hW+fP,fH);
  ef s="l": dwv Z_g--Z_f; dwh Z_s--Z_r; dwh Z_x--Z_z;
  ef s="2": cdw (hP,1.3hP)..(.4fW,.35fH)..(fW,.65aH)..Z_s..(hP,.65aH); dwh Z_d--Z_a;
  ef s="3": cdw sbp(0,.75)circ_Oh; cdw sbp(.25,.95)circ_sOh shifted (.2hP,hH-hP);
            dwh (.3aW,hH)--(.7aW,hH);
  ef s="4": dwh Z_j--Z_k; dwv Z_l--(0.75aW,aH)--(1.2hP,qH+hP); dwv (.75aW+qP,aH)--(1.7hP,qH+hP);
  ef s="-": dwh Z_m--Z_o;
  ef s="+": dwv Z_x--Z_s; dwh Z_m--Z_o;
  ef s="A": dwvs(1.14)Z_b--Z_f--Z_e; dw .33[Z_b,Z_f]--.33[Z_e,Z_f];
  ef s="B": dw Z_r--Z_s{right}..(.85fH,.75aH)..{left}Z_y--Z_m--Z_y{right}..(.9fH,qH)..
              {left}Z_x--Z_c; dwv Z_b--Z_w;
  ef s="D": dw Z_r--Z_s..Z_o..Z_x--Z_c; dwv Z_b--Z_w;
  ef s="E": pickup pensquare scaled fP; dw Z_z--Z_c--Z_r--Z_n; dw Z_m--Z_o;
  ef s="G": cdw sbp(.06,.97)circ_O; dwh bot Z_y-- bot Z_v;
  ef s="J": cdw Z_m..(hP,.4aH){down}..{right}Z_x{right}..{up}(fW,.4aH)..Z_t;
  ef s="K": cdw Z_b--Z_w; cdw .35[.45[Z_b,Z_w],Z_u]--Z_e; cdw .35[Z_b,Z_w]--Z_u;
  ef s="L": dwh Z_d--Z_a; dwv Z_b--Z_w;
  ef s="M": dwv Z_b--Z_w; dwvs(1.14)Z_w--Z_x--Z_t; dwv Z_t--Z_e;
  ef s="Q": dw circ_O; dw (.6aW,.4aH)--Z_e;
  ef s="R": dwv Z_b--Z_w; dw Z_r--(.65aW,fH){right}..(fW,.7aH)..{left}(.65aW,.44aH)..(hP,.44aH);
            cdw Z_e{up}..{left}(hW,.44aH);
  ef s="T": dwh Z_q--Z_u; dwv .5[Z_q,Z_u]--Z_g;
  ef s="U": cdw Z_w..Z_m{down}..{right}Z_x{right}..{up}Z_o..Z_t;
  ef s="V": dwvs(1.2)Z_w--Z_g--Z_t;
  ef s="W": dwvs(1.08)Z_w--(aW/4,0)--Z_f--Z_l--Z_t;
  ef s="X": dwvs(1.4)Z_w..Z_e; dwvs(1.4) Z_b..Z_t;
  ef s="Y": dwvs(1.2)Z_w--Z_y--Z_t; dwv Z_y--Z_g;
  ef s="Z": dwh Z_q--Z_u; dwvs(1.4)(1.4hP,fP)--(aW-1.4hP,aH-fP); dwh Z_a--Z_d;
  ef s="a": dw Z_x..Z_o..Z_s..Z_m..cycle; dwv Z_e--Z_t;
  ef s="b": dw Z_x..Z_o..Z_p..Z_m..cycle; dwv Z_b--(hP,1.3aH)
  ef s="c": cdw sbp(.06,.94)Z_o..Z_s..Z_m..Z_x..cycle;
  ef s="d": dw Z_x..Z_o..Z_p..Z_m..cycle; dwv Z_e--(fW,1.3aH);
  ef s="e": cdw sbp(0,.92)Z_o..Z_s..Z_m..Z_x..cycle; dw Z_o--Z_m;
  ef s="f": cdw (.4fW,0)--(.4fW,.75aH){up}..(.75aW,fH)..{down}(fW,.8aH); dwh Z_h--Z_v;
  ef s="g": dw circ_Oa; dw sbp(0,.5)circ_Oh shifted (0,-.5fH); cdw (fW,.7aH)--(fW,-qH);
  ef s="h": cdw Z_b..(hP,.3aH){up}..(hW,.7fH)..{down}(fW,.3aH)..Z_e; dwv (hP,.3aH)--Z_w;
  ef s="i": dwv Z_g--(hW,.7aH); ppcs 1.4fP; dw Z_s;
  ef s="j": cdw (fW,.7aH)--Z_z..(aW/4,-.66fP)..Z_c; ppcs 1.4fP; dw Z_n;
  ef s="k": dwv Z_b--(hP,1.3fH); cdw .5[Z_b,Z_w]--Z_e; cdw .5[Z_b,Z_w]--Z_u;
  ef s="m": cdw Z_b..(hP,.3aH){up}..(.28aW,fH)..{down}(hW,.3aH)..Z_g;
            cdw (hW,.6aH){up}..(.7aW,fH)..{down}(fW,.6aH)..Z_e; dwv (hP,.3aH)--Z_w;
  ef s="n": cdw Z_b{up}..(hW,.8fH)..{down}Z_o..Z_e; dwv Z_b--(hP,.8aH);
  ef s="o": dw Z_x..Z_o..Z_s..Z_m..cycle;
  ef s="p": dw Z_x..Z_o..Z_s..Z_m..cycle; dwv Z_w--(hP,-.3aH);
  ef s="q": dw Z_x..Z_o..Z_s..Z_m..cycle; dwv Z_t--(fW,-.3aH);
  ef s="r": cdw (sbp(.35,.72)Z_x..Z_o..Z_s..Z_m..cycle) shifted(0,-1.5hP); dwv Z_b--Z_w;
  ef s="s": cdw sbp(.05,.45)circ_O; cdw sbp(.55,.95)circ_O; dw (fW,.3aH){up}..{up}(hP,.7aH);
  ef s="t": dwv Z_g--Z_f; dwh (0,.66aH)--(aW,.66aH);
  ef s="u": cdw Z_w..(hP,.55aH){down}..Z_x..(fW,.55aH){up}..Z_t; dwv Z_t--Z_e;
  ef s="v": dwv Z_w--Z_g--Z_t;
  ef s="w": dwv Z_w--(aW/4,0)--Z_f--Z_l--Z_t;
  ef s="x": dwvs(1.4)Z_w--Z_e; dwvs(1.4) Z_t--Z_b;
  ef s="y": dwvs(1.4)(Z_w--Z_y) shifted (0,-.3aH); dwvs(1.4)(Z_t--Z_b) shifted (0,-.3aH);
  ef s="z": dwh Z_q--Z_u; dwvs(1.4)(1.4hP,fP)--(aW-1.4hP,aH-fP); dwh Z_a--Z_d;
  ef s="0": dw Z_m...Z_s...Z_o...Z_x...cycle;
  ef s="1": dwv Z_g--(hW,aH-.3hP)--(hW-fP,aH-fP)--(hW-fP,aH-1.5fP);
  ef s="5": dwh Z_q--Z_u-(hP,0); dwv Z_r--(hP,.55fH);
            cdw (qP,.25aH)..(.65aW,1.3hP)..(fW,.4aH)..(hW,.63aH)..(.7hP,.56aH);
  ef s="6": dw Z_x..(fW,.5fW)..Z_i..(hP,.5fW)..cycle; cdw (.8fP,hH)--Z_f;
  ef s="7": dwh (0,.fH)--Z_u; dwvs(1.2)(aW-1.2hP,aH-fP)--(.4aW,0);
  ef s="8": dw circ_Oh; dw (hP,.75aH)...Z_s...(fW,.75aH)...Z_y...cycle;
  ef s="9": dw (Z_x..(fW,.5fW)..Z_i..(hP,.5fW)..cycle) shifted (0,.32aH); cdw (fW-.45fP,hH)--Z_g;
  else:
  fi
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def warn_valence(expr a)=
  if addA[a]=0:
    warning_s[incr warning_cnt]:="!Strange valence "&str_tbl[strA[a]]&"(A"&decimal(a)&") has"&
                                 fdr(2)(bond_cntA)&" bonds";
    message "["&decimal(fig_num)&"]"&warning_s[warning_cnt]; message ""; fi
enddef;
%-------------------------------------------------------------------------------------------------
vardef erase_char(expr s)=
  temps:=""; nA:=length(s);
  if (nA>=4)and(s<>"COOH"):
    for i=1 upto nA: tempc:=subc(s,i); if scan_c(tempc,"{}_^+")=0: temps:=temps&tempc; fi endfor
    temps
  else: s fi
enddef;
%-------------------------------------------------------------------------------------------------
vardef forbidden_to_underbar(expr s)=
  temps:=""; nA:=length(s);
  for i=1 upto nA:
    tempc:=subc(s,i); if scan_c(tempc,forbiddens)>0:temps:=temps&"_"; else:temps:=temps&tempc; fi
  endfor temps
enddef;
%=================================================================================================
def proc_report_out=
  file_output:=jobname&"-report.txt";
  block_cnt:=split_str(mc,",")(block_s);
  puts "===========================================================================";
  puts " Fig["&decimal(fig_num)&"],Name["& EN&"],Category["&CAT&"],File["&file_input&"]";
  puts "---------------------------------------------------------------------------";
  puts " Row["&decimal(mc_row)&"],Length["&decimal(length(mc))&"],Block"&
         "["&decimal(block_cnt)&"]"&",BackboneA["&decimal(backboneA)&"]"&
         ",BackboneB["&decimal(backboneB)&"]"&",Group["&decimal(grp_num)&"]";
  puts "---------------------------------------------------------------------------";
  for i=1 upto mc_row: puts (substring(0,mc_indent[i]) of blanks)&mc[i]; endfor
  puts "---------------------------------------------------------------------------";
  for i=1 upto block_cnt: puts "["&fdr(3)(i)&"]  "&block_s[i]; endfor
  puts "- error , warning ---------------------------------------------------------";
  for i=1 upto error_cnt:
    puts error_s[error_e[i]]&" ["&decimal(error_b[i])&"] "&block_s[error_b[i]];
  endfor
  for i=1 upto warning_cnt: puts warning_s[i]; endfor
  puts "---------------------------------------------------------------------------";
  puts " Width["&fdr(8)(mol_wd)&"],Height["&fdr(8)(mol_ht)&"],"&
         " Shift x["& fdr(8)(minX)&"],Shift y["&fdr(8)(minY)&"]";
  puts " Bond length["&fdr(8)(blen)&"],Atom size["&fdr(8)(atom_wd)&"]";
  puts "---------------------------------------------------------------------------";
  puts " Atom["&decimal(cntA)&"],Bond["&decimal(cntB)&
         "],Ring["&decimal(cntB-cntA+1)&"],Hide H["&decimal(hideH_cnt)&"]";
  puts "< NO. ><atom(s) >(  x axis   ,   y axis   )<bond><hideH><chg>";
  for i=1 upto cntA:
    puts " A"&fdl(6)(i)&fsl(8)(erase_char(str_tbl[strA[i]]))&
           " ("&fdr(10)(round(xpart(posA[i])/blen))&" , "&
            fdr(10)(round(ypart(posA[i])/blen))&" ) "&fdr(4)(Bcnt[i])&
            iif(hideH[i]>0,fdr(6)(hideH[i]),"        ") if chargeA[i]<>0: &fdr(4)(chargeA[i]) fi;
  endfor
  puts "---------------------------------------------------------------------------";
  puts "< NO. ><  bond   (sdt)><angle +(  +-  )><length (   pt   )>";
  for i=1 upto cntB:
    nC:=lenB[i]; if nC=_size_atom: nC:=ratio_atom_bond; ef nC<0: nC:=-nC; fi
    nB:=angB[i]; if nB>180: nB:=nB-360; fi
    puts " B"&fdl(4)(i)&fdr(3)(sB[i])&" -> "&fdr(3)(eB[i])&
           " ("&fdr(3)(bond_num[i])&")"&fdr(8)(round(angB[i]))&
           " ("&fdr(6)(round(nB))&")"&fdr(8)(nC)&" ("&fdr(8)(round(nC*blen))&")";
  endfor
  puts "---------------------------------------------------------------------------";
  puts "<atom>( atom wt )[ mi wt   ]  < cnt > < sum wt   >[ sum mi wt  ]";
  for i=1 upto tbl_atom_max:
    if sumA[i]>=1:
      puts " "&fsl(5)(erase_char(tbl_atoms[i]))&
        "("&fdr(9)(tbl_atom_wt[i])&")"&"["&fdr(9)(tbl_atom_mi[i])&"]"&" * "&fdr(8)(sumA[i])
        &"    "&fdr(7)(tbl_atom_wt[i]*sumA[i])&"["&fdr(12)(tbl_atom_mi[i]*sumA[i])&"]"; fi
  endfor
  puts " Molecular Weight [Mono Isotopic] =   "&fsr(12)(mw)&"["&fsr(12)(MI)&"]";
  puts "---------------------------------------------------------------------------";
  puts " Weight Calc: "&mw if MW<>"-": &" - Input: "&MW&" = "&fdr(9)(MW_n-scantokens(MW)) fi;
  puts " Fomula Calc: "&fm if FM<>"-": &" "&iif(fm=FM,"=","<>")&" Data: "&FM fi;
  puts "===========================================================================";
enddef;
%=================================================================================================
def proc_mol_out(expr n)=
  if EN<>"-": EN_:=forbidden_to_underbar(EN); fi
  %-V2000---------------------------------------------------------------------------------------
  if n=2000:
    file_output:="m2k_"&fit_zero(fig_num)&"-"&EN_&".mol";
    puts ""; puts "  -MCFtoMOL- "&fsl(20)(EN); puts "";
    puts fdr(3)(cntA)&fdr(3)(cntB)&"  0  0  0  0  0  0  0  0999 V2000";
    for i=1 upto cntA:
      puts fdr(10)(xpart(posA[i])/blen)& fdr(10)(ypart(posA[i])/blen)&fdr(10)(0)&" "&
           fsl(2)(erase_char(str_tbl[strA[i]]))&"  0"&fdr(3)(bond_charge(chargeA[i]))&"  0  0";
    endfor
    for i=1 upto cntB:
      if lineB[i]<>0: puts fdr(3)(sB[i])&fdr(3)(eB[i])&fdr(3)(bond_type(lineB[i]))&
                      fdr(3)(bond_stereo(lineB[i]))&"     0  0"; fi
    endfor
    puts "M  END"; puts EOF;
  %-V3000---------------------------------------------------------------------------------------
  ef n=3000:
    file_output:="m3k_"&fit_zero(fig_num)&"-"&EN_&".mol";
    puts ""; puts "  -MCFtoMOL- "&fsl(20)(EN); puts "";
    puts "  0  0  0  0  0  0  0  0  0  0  0 V3000"; 
    puts "M  V30 BEGIN CTAB";
    puts "M  V30 COUNTS "&decimal(cntA)&" "&decimal(cntB)&" 0 0 0";
    puts "M  V30 BEGIN ATOM";
    for i=1 upto cntA:
      puts "M  V30 "&decimal(i)&" "&erase_char(str_tbl[strA[i]])&" "&
             decimal(xpart(posA[i])/blen)&" "&decimal(ypart(posA[i])/blen)&" 0 0"
             if chargeA[i]<>0: &" CHG="&decimal(chargeA[i]) fi;
    endfor
    puts "M  V30 END ATOM"; puts "M  V30 BEGIN BOND";
    for i=1 upto cntB:
      if lineB[i]<>0:
        puts "M  V30 "&decimal(i)&" "&decimal(bond_type(lineB[i]))&
               " "&decimal(sB[i])&" "&decimal(eB[i])
               if bond_stereo(lineB[i])<>0: &" CFG="&decimal(bond_config(lineB[i])) fi; fi
    endfor
    puts "M  V30 END BOND"; puts "M  V30 END CTAB"; puts "M  END"; puts EOF; fi
enddef;
%=================================================================================================
def proc_mc_out=
  file_output:="temp-mc.aux";
  if mc_length<100:
    nB:=split_str(mc,",")(arg_s); nA:=0; temps:="";
    for i=1 upto nB:
      if i=nB: temps:=temps&arg_s[i]; puts temps;
      ef at_char[i+1]-nA>mc_length: nA:=at_char[i]; puts temps&arg_s[i]&","; temps:="";
      else: temps:=temps&arg_s[i]&","; fi
    endfor
  else: for i=1 upto mc_row: puts (substring(0,mc_indent[i]) of blanks)&mc[i]; endfor fi
  puts EOF;
enddef;
%=================================================================================================
vardef fit_zero(expr n)= if n<=9: "00" ef n<=99: "0" else: "" fi &decimal(n) enddef;
vardef bond_type(expr n)=
  if (n>=dl)and(n<=dm_):2 ef n=tm:3 ef (n=0)or(n=vf)or(n=vb): 0 else: 1 fi enddef;
vardef bond_charge(expr n)= if n=2: 1 ef n=1: 3 ef n=-1: 5 ef n=-2: 6 else: 0 fi enddef;
vardef bond_stereo(expr n)=
 if (n=wf)or(n=zb)or(n=bd): 1 ef (n=zf)or(n=wb)or(n=dt): 6 ef n=wv: 4 else: 0 fi enddef;
vardef bond_config(expr n)=
 if (n=wf)or(n=zb)or(n=bd): 1 ef (n=zf)or(n=wb)or(n=dt): 3 ef n=wv: 2 else: 0 fi enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef define_atom(expr s,WT,MI)=
  str_cnt:=str_cnt+1; tbl_cnt:=tbl_cnt+1; pc_num:=pc_num+1;
  pc_x[pc_num][1]:=_set_atom; pc_y[pc_num][1]:=str_cnt; pc_cnt[pc_num]:=1;
  str_tbl[str_cnt]:=tbl_atoms[tbl_cnt]:=s; tbl_atom[tbl_cnt]:=0;
  tbl_atom_wt[tbl_cnt]:=WT; tbl_atom_mi[tbl_cnt]:=MI; pc_num
enddef;
%-------------------------------------------------------------------------------------------------
vardef define_grp_string(expr s)(text t)=
  str_cnt:=str_cnt+1; tbl_cnt:=tbl_cnt+1; pc_num:=pc_num+1;
  pc_x[pc_num][1]:=_set_atom; pc_y[pc_num][1]:=str_cnt; pc_cnt[pc_num]:=1;
  str_tbl[str_cnt]:=tbl_atoms[tbl_cnt]:=s; tbl_atom[tbl_cnt]:=0;
  for $=t: tbl_grp[tbl_cnt][incr tbl_atom[tbl_cnt]]:=$-pc_emb_start; endfor pc_num
enddef;
%=================================================================================================
def define_atom_grp_parts=
  begingroup
  save `,``,';
  def `=define_atom enddef; def ``=define_grp_string enddef; def '= define_parts enddef;
  pc_int:=pc_emi_start; pc_num:=pc_emb_start; ?3:=?20:=Ph:=Ph1:=Ph2:=0;
  %-----------------------------------------------------------------------------------------------
  C= `("C"   ,12.0107,   12.0000000);       H= `("H"  , 1.00794,    1.00782503223);
  D= `("D"   ,2.012,      2.01410177812);   Ag=`("{Ag}",107.868,  106.905095);
  Al=`("{Al}",26.9815,   26.98153853);      As=`("{As}",74.9216,   74.92159457);
  B= `("B"   ,10.811,    11.00930536);      Ba=`("{Ba}",137.33,   136.905816);
  Be=`("{Be}",9.01218,   0);                Bi=`("{Bi}",208.9804,  208.980338);
  Br=`("{Br}",79.904,    78.9183376);       Ca=`("{Ca}",40.078,    39.962590863);
  Cd=`("{Cd}",112.41,     110.904182);      Cl=`("{Cl}",35.453,    34.968852);
  Co=`("{Co}",58.933194, 58.93319429);      Cr=`("{Cr}",51.9961,   51.94050623);
  Cs=`("{Cs}",132.905,   132.90543);        Cu=`("{Cu}",63.546,    62.92959772);     
  F= `("F"   ,18.9984,   18.99840316273);   Fe=`("{Fe}",55.845,    55.93493633);
  Hg=`("{Hg}",200.59,    201.97064340);     I= `("I"   ,126.90447,126.9044719);
  K= `("K"   ,39.0983,   38.9637064864);    Li=`("{Li}",6.941,      7.0160034366);
  Mg=`("{Mg}",24.305,    23.985041697);     Mn=`("{Mn}",54.938044, 54.93804391);
  Mo=`("{Mo}",95.95,     0);                N= `("N"   ,14.0067,   14.00307400443);
  Na=`("{Na}",22.98977,  22.9897692820);    Ni=`("{Ni}",58.693,    57.93534241);
  O= `("O"   ,15.9994,   15.99491461957);   P= `("P"   ,30.973762, 30.97376199842);
  Pb=`("{Pb}",207.2,    205.974455);        Pd=`("{Pd}",106.4,    107.905075);
  S= `("S"   ,32.065,    31.9720711744);    Sb=`("{Sb}",121.75,   120.90381);
  Se=`("{Se}",78.971,    79.9165218);       Si=`("{Si}",28.0855,   27.97692653465);
  Sn=`("{Sn}",118.71,   119.90220163);      Ta=`("{Ta}",180.948,   0);
  Te=`("{Te}",127.60,    129.90623);        Ti=`("{Ti}",47.867,    47.94794198);
  U= `("U",   238.0289, 238.05079);         V= `("V",   50.9415,   50.943957);
  W= `("W",   183.85,   181.948225);        Zn=`("{Zn}",65.409,    63.92914201);
  NO_ATOM=`("",0,0);
  tbl_atom_end:=tbl_cnt;
  %-----------------------------------------------------------------------------------------------
  if f_expand=0:
    CH3=``("C{H_3_}")(C,H,H,H); CH2=``("C{H_3_}")(C,H,H); CN=``("CN")(C,N); OH=``("OH")(O,H);
    COOH=``("COOH")(C,O,O,H); COONa=``("COO{Na}")(C,O,O,Na); CHO=``("CHO")(C,H,O);
    NO=``("NO")(N,O); NO2=``("N{O_2_}")(N,O,O); NH2=``("N{H_2_}")(N,H,H); SH=``("SH")(S,H);
    SO2H=``("S{O_2_}H")(S,O,O,H); SO3H=``("S{O_3_}H")(S,O,O,O,H); ONa=``("O{Na}")(O,Na);
    SO3Na:=``("S{O_3_}{Na}")(S,O,O,O,Na); fi
  %-----------------------------------------------------------------------------------------------
  tbl_grp_end:=tbl_cnt; pc_atm_end:=pc_num;
  if f_expand=1:
    NH:='(N,/H); CH3:='(/H,/H^60,/H^-60); NH2:='(N,/H^60,/H^-60); NO2:='(N,//O^60,//O^-60);
    OH:='(O,/H); NO:='(N,//O); SH:='(S,/H); SO2H:='(S,//O^60,/OH^-60);
    SO3H:='(S,//O^90,//O^-90,/OH); CHO:='(//O^-60,/H^60); COOH:='(//O^-60,/OH^60); CN:='(/N~tm);
    ONa:='(O,/Na); SO3Na:='(S,//O^90,//O^-90,/Na);
  else: NH:='(N,/H~nl); fi
  %-----------------------------------------------------------------------------------------------
  !:=!1:='((_mk_bond,_auto_ang)); ?:='(/NO_ATOM); ?wf:=?w:='(/*NO_ATOM); ?zf:=?z:='(*/NO_ATOM);
  ?S:='(//S); ?O:='(//O); ?NH:='(//NH); ?H:='(/H); ?F:='(/F); ?OH:='(/OH); ?OH:='(/COOH);
  ?NH2:='(/NH2); ?Cl:='(/Cl); ?2:='(/!); ?dm:=?d:='((_grp_dm,NO_ATOM));
  for i=3 upto 20: ?[i]:='((_set_len,_ring),<<((-180 DIV i)-90) for j=2 upto i:,360 DIV i endfor
                         ,(_cyc_sB,1-i),(_set_len,_end)); endfor
  for i=2 upto 20: ![i]:='((_set_len,_chain),! for j=2 upto i:,! endfor ,(_set_len,_end)); endfor
  Ph:=Ph1:='(?6,-2==dl,-4==dl,-6==dl); Ph2:='(?6,-1==dl,-3==dl,-5==dl);
  !db:=!d:='(!~db); !tm:=!t:='(!~tm); !wf:=!w:='(!~wf); !zf:=!z:='(!~zf); !wb:='(!~wb);
  !zb:='(!~zb); !dl:='(!~dl); !dr:='(!~dr); !dm:='(!~dm); !d!:='(!~db,!); !!d:='(!,!db);
  !!d!:='(!,!db,!); ?!:='(?,!); !?:='(!,?); ?!d:='(?,!d); ?!2:='(?,!2); ??!:='(?,?^60,60);
  ??:='(?^35,?^-35); !?!:='(!?,!); !??!:='(!,??,!); !??:='(!,??); !?2:='(!,?2); ?2!:='(?2,!);
  n_:='((_add_charge,MIS)); p_:='((_add_charge,PLS));
  mark_adress:='((_com,_mark)); reset_adress:='((_com,_moff)); reset_length:='((_chg_len,_end));
  ?F!F:='(/F,60,F); ?F?F!F:='(/F,/F^60,60,F); ?Br!Br:='(/Br,60,Br); ?Cl!Cl:='(/Cl,60,Cl);
  ?Cl?Cl!Cl:='(/Cl,/Cl^60,60,Cl);
  N!:='(N,!); N?:='(N,?); N!2:='(N,!3); N!2:='(N,!3); !N:='(!,N); N?!:='(N,?!); N?2:='(N,?2);
  S?O:='(S,?O); S?O!:='(S?O,!); ?O?O:='(?O^-35,?O^35); S?O?O:='(S,?O?O); S?O?O!:='(S?O?O,!);
  O!:='(O,!); O!2:='(O,!2); O!3:='(O,!3); S!:='(S,!); S!2:='(S,!2); S!3:='(S,!3); !S:='(!,S);
  ?O!:='(?O,!); ?O!2:='(?O,!2); ?O!3:='(?O,!3); !?O:='(!,?O); !?O!:='(!,?O!); ?O!O:='(?O,!,O);
  NH!:='(NH,!); NH!2:='(NH,!2); !NH:='(!,NH); !NH!:='(!,NH!);
  !OH:='(!,OH); !SH:='(!,SH); !NH2:='(!,NH2); !S?O:='(!,S?O); !O:='(!,O); !dO:='(!d,O);
  !Cl:='(!,Cl); !Br:='(!,Br); !F:='(!,F); !O!:='(!,O!); !S!:='(!,S!);
  !COOH:='(!,COOH); ?COOH:='(/COOH); !CH3:='(!,CH3); !CN:='(!,CN); !CHO:='(!,CHO); !NO2:='(!,NO2);
  !?3:='(!,?3); !?4:='(!,?4); !?5:='(!,?5); !?6:='(!,?6); !?7:='(!,?7); !?8:='(!,?8);
  !Ph:='(!,Ph); !?OH:='(!,?OH); ?OH!:='(?OH,!);
  hexose_hp:='(#1.4,-30~wf_r,30~bd_r`1,30~wb_r,120,O,30,&&1,##);
  Pyranose_hp:='(#1.4,-35~wf_r,35~bd_r`1,30~wb_r,130`1.66,O,&&1,##);
  endgroup
enddef;
%-------------------------------------------------------------------------------------------------
define_atom_grp_parts;
store_restore_par(0)(parameter_list);
for i=pc_emb_start+1 upto pc_num: pc_all:=pc_all+pc_cnt[i]; endfor
message "pc_emb  =" & decimal(pc_emb_start) &" => " & decimal(pc_num) & 
                       " [ " & decimal(pc_all) & " ]";
message "pc_emi  =" & decimal(pc_emi_start) &" => " & decimal(pc_int);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def loadm(text t)=
  begingroup
  save f_mcf,unit_cnt,row_s,row_cnt,info_cnt,order,min_n,max_n,sign_at,sign_n,tag_s,line_cnt,
       val_s,filter_n,filter_tag,filter_val,filter_sign,filter_cnt,filter_p,lib_unit_cnt,at_colon,
       at_equal,at_less,at_greater,at_n,sort_tbl,key_s,sort_s,sign_s,sV,sT,header_s,sort_oder,
       opt_cnt;
  string row_s[][],sort_tbl[],key_s,filter_tag[],filter_val[],sign_s[],tag_s[][],val_s[][],sort_s,
         header_s,sV,sT;
  numeric row_cnt[],order[],order_tbl[],filter_sign[],filter_p[];
  line_cnt:=f_mcf:=mc_row:=info_cnt:=filter_cnt:=lib_unit_cnt:=0; unit_cnt:=1; file_input:=libm;
  sign_s1:="="; sign_s2:="<>"; sign_s3:="<"; sign_s4:=">"; sign_s5:="<="; sign_s6:=">="; 
  header_s:=""; key_s:="";
  %---------------------------------------------------------------------------------------------
  opt_cnt:=0; for $=t: opt_cnt:=opt_cnt+1; endfor
  if opt_cnt=0: filter_cnt:=1; filter_sign[1]=2; filter_tag[1]:="LV"; filter_val[1]:="-";
  else:
    for $=t:
      at_colon:=scan_c(":",$); at_equal:=scan_c("=",$); at_less:=scan_c("<",$);
      at_greater:=scan_c(">",$); key_s:=sT:=sV:=""; sign_n:=sign_at:=at_n:=0;
      if at_colon>=2:                               sign_at:=at_colon;   sign_n:=7; at_n:=1;
      ef at_equal>=2:   if (at_equal-1)=at_less:    sign_at:=at_equal;   sign_n:=5; at_n:=2;
                        ef (at_equal-1)=at_greater: sign_at:=at_equal;   sign_n:=6; at_n:=2;
                        else:                       sign_at:=at_equal;   sign_n:=1; at_n:=1; fi
      ef at_greater>=2: if (at_greater-1)=at_less:  sign_at:=at_greater; sign_n:=2; at_n:=2;
                        else:                       sign_at:=at_greater; sign_n:=4; at_n:=1; fi
      ef at_less>=2:                                sign_at:=at_less;    sign_n:=3; at_n:=1; fi
      if sign_at>=1: sT:=substring(0,sign_at-at_n)of $; sV:=substring(sign_at,length($))of $; fi
      if sign_n=0: if scan_c(".",$)=0: file_input:=$&".mcf"; else: file_input:=$; fi
                   filter_cnt:=1; filter_sign[1]=2; filter_tag[1]:="LV"; filter_val[1]:="-";
      ef sign_n=7: if sT="a": key_s:=sV; sort_oder:=0; ef sT="d": key_s:=sV; sort_oder:=1; fi
      else:
        filter_tag[incr filter_cnt]:=sT; filter_sign[filter_cnt]:=sign_n;
        if is_num(sV): filter_val[filter_cnt]:=fix_num(sV); else: filter_val[filter_cnt]:=sV; fi
      fi
    endfor
  fi
  %--------------------------------------------------------------------------------------------
  forever:
    lines:=readfrom file_input; exitif lines=EOF; firsta:=ASCII(lines);
    if firsta<>PCT:
      row_s[unit_cnt][incr line_cnt]:=lines;
      if firsta=SMC:
        row_cnt[unit_cnt]:=line_cnt; f_mcf:=line_cnt:=0; filter_n:=1; sort_s:="";
        for i=1 upto filter_cnt: filter_p[i]:=0; endfor
        for i=1 upto info_cnt:
          get_tag_var(arg_s[i])(sT,sV); tag_s[unit_cnt][i]:=sT; val_s[unit_cnt][i]:=sV;
          if sT=key_s: if is_num(sV): sort_s:=fix_num(sV); else: sort_s:=sV; fi fi
          for j=1 upto filter_cnt:
            if filter_tag[j]=sT:
              filter_p[j]:=1; if is_num(sV): temps:=fix_num(sV); else: temps:=sV; fi
              if filter_sign[j]=1: if not(temps= filter_val[j]): filter_n:=0; fi
              ef filter_sign[j]=2: if not(temps<>filter_val[j]): filter_n:=0; fi
              ef filter_sign[j]=3: if not(temps< filter_val[j]): filter_n:=0; fi
              ef filter_sign[j]=4: if not(temps> filter_val[j]): filter_n:=0; fi
              ef filter_sign[j]=5: if not(temps<=filter_val[j]): filter_n:=0; fi
              ef filter_sign[j]=6: if not(temps>=filter_val[j]): filter_n:=0; fi fi
            fi
          endfor
        endfor
        for i=1 upto filter_cnt: if filter_p[i]=0: filter_n:=0; fi endfor
        info_cnt:=0; lib_unit_cnt:=lib_unit_cnt+1;
        if filter_n=1: if key_s<>"": sort_tbl[unit_cnt]:=sort_s; fi unit_cnt:=unit_cnt+1; fi
      ef firsta=CLN: f_mcf:=mc_row:=1;
                     unit_info[unit_cnt]:=info_cnt:=split_str(header_s,";")(arg_s); header_s:="";
      ef firsta<>PCT: if f_mcf=1: mc_row:=mc_row+1;
                      else: if header_s="": header_s:=lines; else: header_s:=header_s&lines; fi fi
      fi
    fi
  endfor
  ucount:=unit_cnt:=unit_cnt-1;
  %----------------------------------------------------------------------------------------------
  message "* Input  : "&file_input&" ["&decimal(lib_unit_cnt)&"]";
  message "* Output : ucount ["&decimal(ucount)&"]"; message "";
  for i=1 upto filter_cnt:
    message "* Filter("&decimal(i)&"): "&filter_tag[i]&" "&sign_s[filter_sign[i]]&filter_val[i];
  endfor
  if key_s<>"": message "* Sort key : "&key_s&iif(sort_oder=0," (ascending)"," (descending)"); fi
  if key_s<>"":
    for i=1 upto unit_cnt: order[i]:=0; endfor
    for i=1 upto unit_cnt:
      if sort_oder=0: temps:="~";
        for j=1 upto unit_cnt:
          if order[j]=0: if sort_tbl[j]<temps: temps:=sort_tbl[j]; min_n:=j; fi fi
        endfor
        order[min_n]:=i; order_tbl[i]:=min_n;
      ef sort_oder=1: temps:=" ";
        for j=1 upto unit_cnt:
          if order[j]=0: if sort_tbl[j]>temps: temps:=sort_tbl[j]; max_n:=j; fi fi
        endfor
        order[max_n]:=i; order_tbl[i]:=max_n; fi
    endfor
    for i=1 upto unit_cnt:
      for j=1 upto unit_info[order_tbl[i]]:
        lib_tag[i][j]:=tag_s[order_tbl[i]][j]; lib_val[i][j]:=val_s[order_tbl[i]][j];
      endfor
      unit_lines[i]:=row_cnt[order_tbl[i]];
      for j=1 upto row_cnt[order_tbl[i]]: row[i][j]:=row_s[order_tbl[i]][j]; endfor
    endfor
  else:
    for i=1 upto unit_cnt:
      for j=1 upto unit_info[i]: lib_tag[i][j]:=tag_s[i][j]; lib_val[i][j]:=val_s[i][j]; endfor
      unit_lines[i]:=row_cnt[i]; for j=1 upto row_cnt[i]: row[i][j]:=row_s[i][j]; endfor
    endfor
  fi
  closefrom file_input;
  endgroup; % [;]
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef check_block(expr s)=
  save len_s,err_type,char_cnt,mc_chr,mc_asc; numeric mc_chr[],mc_asc[];
  err_type:=char_cnt:=prs_cnt:=pre_cnt:=0; nA:=1; len_s:=length(s);
  for i=1 upto len_s: mc_chr[i]:=ASCII(subc(s,i)); endfor
  for i=1 upto len_s: if mc_chr[i]<>BLK: mc_asc[incr char_cnt]:=mc_chr[i]; fi endfor
  mc_asc[0]:=mc_asc[char_cnt+1]:=CMA;
  for i=1 upto char_cnt:
    nA:=mc_asc[i-1]; nB:=mc_asc[i]; nC:=mc_asc[i+1];
    if (nB=BRS)or(nB=BRE)or(nB=BKS)or(nB=BKE)or(nB=BSL)or(nB=DQT)
       or(nB=PCT)or(nB=SMC)or(nB=PLS): err_type:=3;                                   % {}[]\"%;+
    ef (nB=EQU)or(nB=CLN)or(nB=HAT)or(nB=BQT)or(nB=TLD)or(nB=LTN)or(nB=QUT):          % =:^`~>'
      if (nA=CMA)or(nC=CMA)or(nA=QUT)or(nC=QUT)or(nC=CLN): err_type:=1; fi
    ef (nB=AMK)or(nB=AMP)or(nB=GTN):                                                  % @&<
      if nA<>CMA: err_type:=2; fi if (nC=CMA)or(nC=QUT): err_type:=1; fi
    ef nB=DLL: if (nC=QUT)or(nC=CMA): err_type:=1; fi                                 % $
    ef nB=SLS: if (nA<>CLN)and(nA<>CMA)and(nA<>AST)and(nA<>SLS): err_type:=2; fi      % /
    ef nB=PRS: prs_cnt:=prs_cnt+1;                                                    % (
    ef nB=PRE: pre_cnt:=pre_cnt+1;                                                    % )
    ef nA=SLS: if (nB=SLS): if (nC=CMA)or(nC=QUT): err_type:=1; fi                    % //
               ef (nB=AST): if (nC=CMA)or(nC=QUT): err_type:=1; fi                    % /*
               ef (nB=CMA)or(nB=QUT):              err_type:=1; fi                    % /
    ef (nA=AST)and(nB=SLS): if (nC=CMA)or(nC=QUT): err_type:=1; fi                    % */
    fi
  endfor
  if prs_cnt<>pre_cnt: err_type:=7; fi
  err_type
enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vardef fix_num(expr s)=
  temps:=s; nN:=scan_c(".",temps);
  if nN=0: temps:=fsr(4)(temps); ef nN=1: temps:="   0"&temps; ef nN=2: temps:="   "&temps;
  ef nN=3: temps:="  "&temps; ef nN=4: temps:=" "&temps; fi temps
enddef;
vardef is_num(expr s)=
  for i=1 upto length(s): tempc:=subc(s,i); bA:=((tempc>="0")and(tempc<="9"))or(tempc=".");
    exitif not bA; endfor bA
enddef;
vardef scan_bit(expr n,b)= if b>=1: odd(floor(n/b)) else: odd(floor((frac n)/b)) fi enddef;
vardef scan_char(expr c,s,d,n)=
  nN:=0; nA:=length(s);
  if d=0:  for i=n upto nA:   if subc(s,i)=c:  nN:=i; fi exitif nN>0; endfor
  ef d=1:  for i=n upto nA:   if subc(s,i)<>c: nN:=i; fi exitif nN>0; endfor
  ef d=-1: for i=nA downto n: if subc(s,i)<>c: nN:=i; fi exitif nN>0; endfor fi nN
enddef;
vardef scan_c(expr c,s)=
  nN:=0; nA:=length(s); for i=1 upto nA: if subc(s,i)=c: nN:=i; fi exitif nN>0; endfor nN
enddef;
vardef split_str(expr s,c)(suffix v)=
  at_char[0]:=nN:=0; nA:=length(s);
  for i=1 upto nA: if subc(s,i)=c: at_char[incr nN]:=i; fi endfor at_char[incr nN]:=nA+1;
  for i=1 upto nN: v[i]:=substring (at_char[i-1],at_char[i]-1) of s; endfor nN
enddef;
vardef replace_str(expr s,c,cc)=
  nN:=0; nA:=length(s); for i=1 upto nA: if subc(s,i)=c: at_char[incr nN]:=i; fi endfor
  at_char[incr nN]:=nA+1; temps:=substring (0,at_char[1]-1) of s;
  for i=2 upto nN: temps:=temps&cc&substring (at_char[i-1],at_char[i]-1) of s; endfor temps
enddef;
vardef get_tag_var(expr s)(suffix t,v)=
  nN:=scan_c(":",s); t:=substring(0,nN-1) of s; v:=substring(nN,length(s)) of s;
enddef;
def allm= for i=1 upto ucount: beginfigm getm(i) putm endfigm endfor enddef;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
