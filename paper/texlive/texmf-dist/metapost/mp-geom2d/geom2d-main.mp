

%%% ----------------------------------------------------------------------------
%%% Déclarations
%%% ----------------------------------------------------------------------------

numeric gddA[];
numeric gddB[];
numeric gddC[];
numeric gddD[];
numeric gddE[];
numeric gddF[];
numeric gddX[][]; 
picture gddL[];         % Table des labels TeX (btex .. etex)
numeric gddO;           % Compteur des objets
path    gddP[];         % Table des chemins (paths internes)
path    gddPX[][];      % Tables des chemins étendues 
string  gddS[];         % Table des chaînes (noms de fichiers)
string  gddT[];         % Table des types
numeric gddU;           % Unité graphique
string  gddW;           % Chaîne spécifiant le placement des objets

%%% ----------------------------------------------------------------------------
%%% Initialisations
%%% ----------------------------------------------------------------------------

% Initialisation de gddO -------------------------------------------------------
gddO := 0;

% Valeur par défaut de gddU ----------------------------------------------------
gddU := 1cm;

% Initialisation de gddW et procédure associée ---------------------------------
gddW = "scaled gddU";
def gddEnPlace = scantokens gddW enddef;

% anglais
let gddInPlace = gddEnPlace;

%%% ----------------------------------------------------------------------------
%%% Définition des objets
%%% ----------------------------------------------------------------------------

% Point ------------------------------------------------------------------------
vardef Point(expr a,b) =
  gddT[incr gddO] = "point";
  gddA[gddO] = a; gddB[gddO] = b; gddO
enddef;

vardef PointPolaire(expr r,a) =
  % r : rayon
  % a : angle en radian
  gddT[incr gddO] = "point";
  gddA[gddO] = r*cos(a); gddB[gddO] = r*sin(a); gddO
enddef;

% anglais
vardef PolarPoint(expr r,a) = PointPolaire(r,a) enddef;

vardef PointDansRepere(expr x,y,o,i,j) =
  gddT[incr gddO] = "point";
  gddA[gddO] = gddA[o] + x*(gddA[i]-gddA[o]) + y*(gddA[j]-gddA[o]);
  gddB[gddO] = gddB[o] + x*(gddB[i]-gddB[o]) + y*(gddB[j]-gddB[o]);
  gddO
enddef;

% anglais
vardef PointInBasis(expr x,y,o,i,j) = PointDansRepere(x,y,o,i,j) enddef;

vardef CoordonneesRepere(expr m,o,i,j) =
  save x,y; numeric x,y;
  gddA[m] = (gddA[i]-gddA[o])*x + (gddA[j]-gddA[o])*y + gddA[o];
  gddB[m] = (gddB[i]-gddB[o])*x + (gddB[j]-gddB[o])*y + gddB[o];
  (x,y)
enddef;

% anglais
vardef CoordinatesInBasis(expr m,o,i,j) = CoordonneesRepere(m,o,i,j) enddef;

vardef Milieu(expr a,b) =
  gddT[incr gddO] = "point";
  gddA[gddO] = 0.5*(gddA[a]+gddA[b]);
  gddB[gddO] = 0.5*(gddB[a]+gddB[b]);
  gddO
enddef;

%anglais
vardef MidPoint(expr a,b) = Milieu(a,b) enddef;

vardef Rotation(expr A, a) = 
  % A : point
  % a : angle
  save _tmp,n;
  n = incr gddO;
  _tmp = PointImp(A);
  gddT[n] = "point";
  gddA[n] = cos(a)*gddA[_tmp]-sin(a)*gddB[_tmp];
  gddB[n] = sin(a)*gddA[_tmp]+cos(a)*gddB[_tmp];
  n
enddef;  



vardef RotationCentre(expr A, C, a) = 
  % A : point
  % C : point centre de la rotation
  % a : angle
  save _tmp,_tmp,n;
  n = incr gddO;
  _tmp1 = PointImp(A);
  _tmp2 = PointImp(C);
  gddT[n] = "point";
  gddA[n] = cos(a)*(gddA[_tmp1]-gddA[_tmp2])-sin(a)*(gddB[_tmp1]-gddB[_tmp2])+gddA[_tmp2];
  gddB[n] = sin(a)*(gddA[_tmp1]-gddA[_tmp2])+cos(a)*(gddB[_tmp1]-gddB[_tmp2])+gddB[_tmp2];
  n
enddef;

%anglais
vardef CenterRotation(expr A, C, a) = RotationCentre(A,C,a) enddef;

vardef IsoBarycentre(text t) =
  save n, G;
  pair G; numeric n;
  G := origin; n:=0;
  for a=t:
    G:=G+PairImp(a);
    n:=n+1;
  endfor;
  PointImp(G/n)
enddef;

vardef IsoBarycenter(text t) = IsoBarycentre(t) enddef;

vardef Barycentre(text t) =
  % liste sous la forme (A,a),(B,b),(C,c), etc.
  save n, G;
  pair G; numeric n;
  G := origin; n:=0;
  for a=t:
    G:=G+((ypart a)*Pt(xpart a));
    n:=n+(ypart a);
  endfor;
  PointImp(G/n)
enddef;

vardef Barycenter(text t) = Barycentre(t) enddef;

% Transformations pair <-> Point -----------------------------------------------
def PointTOPair(expr a) = (gddA[a],gddB[a]) enddef;
def PairTOPoint(expr p) = Point(xpart p,ypart p) enddef;
def PairImp(expr a) = if pair a: a else: PointTOPair(a) fi enddef;
def PointImp(expr a) = if pair a: PairTOPoint(a) else: a fi enddef;
def Pt(expr a) = (gddA[a],gddB[a]) enddef;
def PtR(expr a) = (gddA[a],gddB[a]) gddEnPlace enddef;

% Transformations pair <-> Point -----------------------------------------------
def VecteurTOPair(expr a) = (gddA[a],gddB[a]) enddef;
def PairTOVecteur(expr p) = Vecteur(xpart p,ypart p) enddef;

% Abscisse et ordonnées --------------------------------------------------------
vardef Abscisse(expr a) = if pair a: xpart a else: gddA[a] fi enddef;
vardef Ordonnee(expr a) = if pair a: ypart a else: gddB[a] fi enddef;

vardef Xcoordinate(expr a) = Abscisse(a) enddef;
vardef Ycoordinate(expr a) = Ordonnee(a) enddef;

% Addition ---------------------------------------------------------------------
vardef AdditionAbscisses(expr a,b) = if pair a: xpart a else: gddA[a] fi + if pair b: xpart b else: gddA[b] fi enddef;
vardef AdditionOrdonnees(expr a,b) = if pair a: ypart a else: gddB[a] fi + if pair b: ypart b else: gddB[b] fi enddef;
vardef Addition(expr a,b) =
  gddT[incr gddO] = "point";
  gddA[gddO] = AdditionAbscisses(a,b);
  gddB[gddO] = AdditionOrdonnees(a,b);
  gddO
enddef;

% anglais 
vardef XcoordinateSum(expr a,b) = AdditionsAbsicces(a,b) enddef;
vardef YcoordinateSum(expr a,b) = AdditionOrdonnees(a,b) enddef;
vardef PointSum(expr a,b) = Addition(a,b) enddef;

vardef Longueur(expr a,b) = 
  abs(PairImp(a)-PairImp(b))
enddef;

vardef Length(expr a,b) = Longueur(a,b) enddef;

vardef Bissectrice(expr a,b,c) = 
  % bisectrice du secteur angulaire défini par les points (abc)
  save dba,dbc,ap,cp;
  dba = Droite(b,a);
  dbc = Droite(b,c);
  ap = ReportSurDroite(b,dba,2);
  cp = ReportSurDroite(b,dbc,2);
  %% C'est une bonne idée cette procédure ReportSurDroite !
  %% Pour obtenir la bissectrice il suffit de prendre la droite qui joint
  %% le sommet au milieu du segment dont les extrémités sont les reports.
  %% Ainsi le cas où les droites (b,a) et (b,c) sont confondues est traité.
  Droite(b,Milieu(ap,cp))
enddef;

vardef AngleBisector(expr a,b,c) = Bissectrice(a,b,c) enddef;

vardef PointIntersection(expr p,q)=
  save A;
  pair A;
  A := (gddTraceObjet(p)) intersectionpoint (gddTraceObjet(q));
  PointImp(A)
enddef;

vardef IntersectionPoint(expr p,q)= PointIntersection(p,q) enddef;


% Vecteur ----------------------------------------------------------------------
vardef Vecteur (expr a,b) =
  save n; n = incr gddO;
  gddT[n] = "vecteur"; gddA[n] = a; gddB[n] = b; n
enddef;

vardef Vector(expr a,b) = Vecteur(a,b) enddef;

vardef VecteurP (expr a) =
  % définition d’un vecteur à partir d’un pair
  save n; n = incr gddO;
  gddT[n] = "vecteur"; gddA[n] =xpart a; gddB[n] = ypart a; n
enddef;

vardef VectorP(expr a) = VecteurP(a) enddef;

vardef ScalaireVecteur (expr k,a) =
  % multiplication par un scalaire
  % k : scalaire
  % a : vecteur
  save x,y;
  x = k*gddA[a];
  y = k*gddB[a];
  Vecteur(x,y)
enddef;

vardef ScalarVector(expr k,a) = ScalaireVecteur(k,a) enddef;

vardef AdditionVecteur (expr u,v) = 
  save n,x,y; n = incr gddO;
  gddT[n] = "vecteur";
  x = gddA[u]+gddA[v];
  y = gddB[u]+gddB[v];
  gddA[n] = x; gddB[n] = y; n
enddef;

vardef VectorSum(expr u,v) = AdditionVecteur(u,v) enddef;

vardef SoustractionVecteur (expr u,v) = 
  save n,x,y; n = incr gddO;
  gddT[n] = "vecteur";
  x = gddA[u]-gddA[v];
  y = gddB[u]-gddB[v];
  gddA[n] = x; gddB[n] = y; n
enddef;

vardef VectorSubstract(expr u,v) = SoustractionVecteur(u,v) enddef;

vardef ProduitScalaire (expr u,v)=
  (gddA[u]*gddA[v]+gddB[u]*gddB[v])
enddef;

vardef ScalarProduct(expr u,v) = ProduitScalaire(u,v) enddef;

vardef Norme (expr s) =
  sqrt(gddA[s]**2+gddB[s]**2) 
enddef;

vardef Norm(expr s) = Norme(s) enddef;

vardef VecteursAngle (expr u,v)=
  arccos((ProduitScalaire(u,v))/(Norme(u)*Norme(v)))
enddef;

vardef AngleBetweenVectors(expr u,v) = VecteursAngle(u,v) enddef;

% Segment ----------------------------------------------------------------------
vardef Segment (expr a,b) =
  save n; n = incr gddO;
  gddT[n] = "segment"; gddA[n] = PointImp(a); gddB[n] = PointImp(b); n
enddef;

vardef LongueurSegment (expr s) =
  abs(Pt(gddA[s])-Pt(gddB[s]))  
enddef;

vardef SegmentLength(expr s) = LongueurSegment(s) enddef;

vardef SegmentTOVecteur (expr s)=
  VecteurP(Pt(gddB[s])-Pt(gddA[s]))
enddef;

vardef SegmentTOVector(expr s) = SegmentTOVecteur(s) enddef;

% Triangle ---------------------------------------------------------------------
vardef Triangle (expr a,b,c) =
  save n; n = incr gddO; gddT[n] = "triangle";
  gddA[n] = PointImp(a); gddB[n] = PointImp(b); gddC[n] = PointImp(c); n
enddef;

% Drawing with Metapost, Thurston
vardef Orthocentre (expr t) =
  save A,B,C,D,p,q;
  pair A,B,C,D,p,q;
  A = PointTOPair(gddA[t]);
  B = PointTOPair(gddB[t]);
  C = PointTOPair(gddC[t]);
  p = whatever[B,C] ; A-p = whatever*(B-C) rotated 90;
  q = whatever[C,A] ; B-q = whatever*(C-A) rotated 90;
  D = whatever[A,p] = whatever[B,q];
  PairTOPoint(D)
enddef;

vardef Orthocenter(expr t) = Orthocentre(t) enddef;

vardef AireTriangle ( expr t) = 
  save droite,h,l;
  droite = Droite(gddA[t],gddB[t]);
  h = ProjectionPointSurDroite(gddC[t],droite);
  l = Longueur(gddC[t],h);
  (Longueur(gddA[t],gddB[t])*l/2.0)
enddef;

vardef TriangleArea(expr t) = AireTriangle(t) enddef;

vardef CercleInscrit(expr t) =
  save AB,BC,CA,a,b,c,I,Aire,rayon;
  c = Longueur(gddA[t],gddB[t]);
  a = Longueur(gddB[t],gddC[t]);
  b = Longueur(gddC[t],gddA[t]);
  I = Barycentre((gddA[t],a),(gddB[t],b),(gddC[t],c));
  Aire = AireTriangle(t);
  rayon = 2*Aire/(a+b+c);
  Cercle(I,rayon)
enddef;

vardef InscribedCircle(expr t) = CercleInscrit(t) enddef;

vardef CercleExinscrit(expr t,n) =
  % t : gdd triangle
  % s : 1,2 ou 3 pour choisir AB, BC ou CA
  save AB,BC,CA,a,b,c,I,p,Aire,rayon;
  c = Longueur(gddA[t],gddB[t]);
  a = Longueur(gddB[t],gddC[t]);
  b = Longueur(gddC[t],gddA[t]);
  p = (a+b+c)/2.0;
  Aire = AireTriangle(t);
  if(n=1):
    rayon = Aire/(p-a);
    I = Barycentre((gddA[t],-a),(gddB[t],b),(gddC[t],c));  
  elseif(n=2):
    rayon = Aire/(p-b);
    I = Barycentre((gddA[t],a),(gddB[t],-b),(gddC[t],c));
  elseif(n=3):
    rayon = Aire/(p-c);
    I = Barycentre((gddA[t],a),(gddB[t],b),(gddC[t],-c));
  fi  
  Cercle(I,rayon)
enddef;

vardef EscribedCircle(expr t,n) = CercleExinscrit(t,n) enddef;

vardef CercleCirconscrit(expr t) =
  % t : gdd triangle
  CercleTroisPoints(gddA[t],gddB[t],gddC[t])
enddef;

vardef CircumscribedCircle(expr t) = CercleCirconscrit(t) enddef;

vardef CercleEuler(expr t) =
  % t : gdd triangle
  % cercle des neuf points 
  save D,C,N,r;
  pair N;
  D = Orthocentre(t);
  C = CercleCirconscrit(t);
  N = 0.5[Pt(D),Pt(Centre(C))];
  r = abs(N-1/2[Pt(gddA[t]),Pt(gddB[t])]); % un des neuf points est le
                                               % milieu de [A,B]
  Cercle(N,r)  
enddef;

vardef EulerCircle(expr t) = CercleEuler(t) enddef;

% Polygones --------------------------------------------------------------------
vardef Polygone (text plist) =
  save n,_point,i; n = incr gddO; gddT[n] = "polygone";
  i:=1;
  for _point = plist:
    gddX[gddO][i] = PointImp(_point);
    i:=i+1;
  endfor
  gddB[n] = i-1; % nombre de côté
  gddA[n] = IsoBarycentre(plist); % centre
  n
enddef;

vardef Polygon(text plist) = Polygone(plist) enddef;

vardef PolygoneRegulier (expr N,s,r,t)=
  % N : (entier) nombre de côtés
  % s : (numeric) rayon du cercle circonscrit
  % r : (numeric en degré) rotation du polygone (premier point 
  %      commence sur l’axe des abscisses)
  % t : (pair ou point) translation 
  save n,_point,i; n = incr gddO; gddT[n] = "polygone";
  pair _point;
  for i=1 upto N:
    _point := (s*cos((2*Pi/N)*(i-1)+r/180*Pi),s*sin((2*Pi/N)*(i-1)+r/180*Pi))+PairImp(t);
    gddX[n][i] = PointImp(_point);
  endfor
  gddB[n] = N; % nombre de côtés
  gddA[n] = PointImp(t); % centre
  n
enddef;

vardef RegularPolygon(expr N,s,r,t) = PolygoneRegulier(N,s,r,t) enddef;

def NombreCotesPolygone(expr p) = 
  gddB[p]
enddef;

vardef PolygonNumberOfSides(expr p) = NombreCotesPolygone(p) enddef;

def PointPolygone(expr p,i) =
  gddX[p][i]
enddef;

def PolygonPoint(expr p,i) = PointPolygone(p,i) enddef;

% Droite -----------------------------------------------------------------------
vardef Droite (expr a,b) =
  save n; n = incr gddO;
  gddT[n] = "droite"; gddA[n] = PointImp(a); gddB[n] = PointImp(b); n
enddef;

vardef Line(expr a,b) = Droite(a,b) enddef;

% Équation de droite : triplet (u,v,w) tel que ux+vy+w=0 -----------------------
% L'équation rendue est normalisée !
vardef EquationDroite(expr d) =
  save u, v, w, n;
  u := gddB[gddB[d]] - gddB[gddA[d]];
  v := gddA[gddA[d]] - gddA[gddB[d]];
  w := - u * gddA[gddA[d]] - v * gddB[gddA[d]];
  n := u ++ v;
  (u/n,v/n,w/n)
enddef;

vardef LineEquation(expr d) = EquationDroite(d) enddef;

% Ordonnée relative à une droite orientée --------------------------------------
vardef OrdonneeRelativePointDroite(expr p, d) =
  savec u, v, w;
  (u,v,w) = EquationDroite(d);
  u * gddA[p] + v * gddB[p] + w
enddef;

vardef PointLineRelativeDistance(expr p,d) = OrdonneeRelativePointDroite(p,d) enddef;

% Distance d'un point à une droite ---------------------------------------------
vardef DistancePointDroite(expr p, d) =
  abs(OrdonneeRelativePointDroite(p, d))
enddef;

vardef PointLineDistance(expr p,d) = DistancePointDroite(p,d) enddef;

% Projection d'un point sur une droite -----------------------------------------
vardef ProjectionPointSurDroite(expr p, d) =
  save u,v,w, l;
  (u,v,w) = EquationDroite(d);
  l := - u * gddA[p] - v * gddB[p] - w;
  Point(gddA[p] + l * u, gddB[p] + l * v)
enddef;

vardef PointOnLineProjection(expr p,d) = ProjectionPointSurDroite(p,d) enddef;

% Droite perpendicalaire -------------------------------------------------------

vardef DroitePerpendiculaire (expr D,p) =
  % D : droite par rapport à laquelle on doit être perpendiculaire
  % p : point par lequel doit passer la droite
  save n,_t,A,B,v,pP,pD; n = incr gddO;
  pair _t,A,B,v,q;
  A := Pt(gddA[D]);
  B := Pt(gddB[D]);
  v := B-A; v := (-(ypart v), xpart v); %% Vecteur normal à B-A
  q := Pt(p) + v;
  Droite(p,q)
enddef;

vardef PerpendicularLine(expr D,p) = DroitePerpendiculaire(D,p) enddef;

% Droite parallele -------------------------------------------------------

vardef DroiteParallele (expr D,p) =
  % D : droite par rapport à laquelle on doit être parallele
  % p : point par lequel doit passer la droite
  save n,_t,A,B,v,pP,pD; n = incr gddO;
  pair _t,A,B,v,q;
  A := Pt(gddA[D]);
  B := Pt(gddB[D]);
  v := B-A;  %% Vecteur de la droite D
  q := Pt(p) + v;
  Droite(p,q)
enddef;

vardef ParallelLine(expr D,p) = DroiteParallele(D,p) enddef;

vardef ReportSurDroite(expr P,D,d) = 
  % P : point de départ du segment
  % D : droite sur laquelle construire le point
  % d : longueure du segment
  save n,_t; n = incr gddO;
  pair _t;
  _t := PairImp(P)+d*unitvector(Pt(gddB[D])-Pt(gddA[D]));
  gddT[n] = "point"; gddA[n] = xpart _t; gddB[n] = ypart _t; n
enddef;

vardef ReportOnLine(expr P,D,d) = ReportSurDroite(P,D,d) enddef;

% Intersection de deux droites -------------------------------------------------
vardef IntersectionDroites (expr a,b) =
  save _p;
  pair _p;
  _p :=  mpIntersectionDroites(Pt(gddA[a]),Pt(gddB[a]),Pt(gddA[b]),Pt(gddB[b]));
  PairTOPoint(_p)
enddef;

vardef LinesIntersection(expr a,b) = IntersectionDroites(a,b) enddef;

% Cercles ----------------------------------------------------------------------
vardef Cercle (expr a,b) =
    save n; n = incr gddO;
    gddT[n] = "cercle"; gddA[n] = PointImp(a); gddB[n] = b; n
enddef;

vardef Circle(expr a,b) = Cercle(a,b) enddef;

vardef Rayon(expr c) = 
  gddB[c]
enddef;

vardef Radius(expr c) = Rayon(c) enddef;

vardef CercleCP(expr a,b) = Cercle(a,abs(Pt(a)-Pt(b))) enddef;

vardef CircleCP(expr a,b) = CercleCP(a,b) enddef;

vardef CercleD(expr a,b) = CercleCP(Milieu(a,b),a) enddef;

vardef CircleD(expr a,b) = CercleD(a,b) enddef;

% Drawing with MetaPost Thurston
vardef CercleTroisPoints(expr A, B, C) =
  save o; pair o;
  o = whatever * (Pt(A)-Pt(B)) rotated 90 shifted 1/2 [Pt(A),Pt(B)] = whatever * (Pt(B)-Pt(C)) rotated 90 shifted 1/2 [Pt(B),Pt(C)];
  Cercle(o,abs(Pt(A)-o))
enddef;

vardef CircleThreePoints(expr A,B,C) = CercleTroisPoints(A,B,C) enddef;

vardef AxeDeSimilitude(expr a,b,c) = 
  % Axe de similitude de trois cercles
  save T,E;
  numeric T[],E[];
  T1 = TangenteCommuneExterieure(a,b);
  T2 = TangenteCommuneExterieure(b,a);
  T3 = TangenteCommuneExterieure(b,c);
  T4 = TangenteCommuneExterieure(c,b);
  E1 = IntersectionDroites(T1,T2);
  E2 = IntersectionDroites(T3,T4);
  Droite(E1,E2)
enddef;

vardef AxisOfSimilitude(expr a,b,c) = AxeDeSimilitude(a,b,c) enddef;

% Tangente commune extérieure à deux cercles -----------------------------------
vardef TangenteCommuneExterieure(expr a, b) =
  save O, A, d, f; pair O, A;
  O = (gddB[a]/(gddB[a]-gddB[b]))[Pt(gddA[a]),Pt(gddA[b])]; % Centre de l'homothétie
  d = abs(O-Pt(gddA[a]));
  f := if gddA[a] < gddA[b] : 1 else: -1 fi;
  A = Pt(gddA[a]) rotatedaround(O,angle(d+-+gddB[a],f*gddB[a]));
  Droite(O,A)
enddef;

vardef ExternalCommonTangent(expr a,b) = TangenteCommuneExterieure(a,b) enddef;

% Tangente commune intérieure à deux cercles -----------------------------------
vardef TangenteCommuneInterieure(expr a, b) =
  save O, A, d, f; pair O, A;
  O = (gddB[a]/(gddB[a]+gddB[b]))[Pt(gddA[a]),Pt(gddA[b])]; % Centre de l'homothétie
  d = abs(O-Pt(gddA[a]));
  f := if gddA[a] < gddA[b] : 1 else: -1 fi;
  A = Pt(gddA[a]) rotatedaround(O,angle(d+-+gddB[a],f*gddB[a]));
  Droite(O,A)
enddef;

vardef InternalCommonTangent(expr a,b) = TangenteCommuneInterieure(a,b) enddef;

% Intersection de deux cercles -------------------------------------------------
vardef IntersectionCercles (expr a, b) =
  save d, x, y, u; pair u;
  u = unitvector(Pt(gddA[b])-Pt(gddA[a]));
  d = abs(Pt(gddA[a])-Pt(gddA[b]));
  x = (gddB[a]**2-gddB[b]**2+d**2)/(2d);
  y = gddB[a]+-+x;
  PairTOPoint(Pt(gddA[a])+x*u+y*(u rotated 90))
enddef;

vardef CirclesIntersection(expr a,b) = IntersectionCercles(a,b) enddef;


% Intersection d’une droite et d’un cercle ------------------------------------
vardef IntersectionDroiteCercle (expr D, CC, n) = 
  % D : droite
  % CC : cercle 
  % n : identification du point 1 ou 2, le même si point unique
  % résolution par le calcul 
  save a,b,c,Xc,Yc,R,A,B,C,Delta,Xp,Yp;
  numeric a,b,c,Xc,Yc,R,A,B,C,Delta,Xp,Yp;
  (a,b,c) = EquationDroite(D);
  (Xc,Yc) = Pt(Centre(CC));
  R := Rayon(CC);
  if(b=0): % si la droite est verticale
    Xp := xpart Pt(gddA[D]);
    A := 1;
    B := 2*(-Yc);
    C := Yc**2+(Xp-Xc)**2-R**2;
    Delta := B**2-4*A*C;
    if(Delta>0):
      if(n=1):
        Yp := (-B - sqrt(Delta))/(2*A);
      elseif(n=2):
        Yp := (-B + sqrt(Delta))/(2*A);
      fi
    elseif(Delta=0):
      Yp := -B/(2*A);
    fi
  else:
    if(abs(a/b)<8): % si la droite est suffisamment horizontale  
      A := 1+(-a/b)**2;
      B := 2*((-a/b)*(-c/b-Yc)-Xc);
      C := Xc**2+(-c/b-Yc)**2-R**2;
      Delta := B**2-4*A*C;
      if(Delta>0):
        if(n=1):
          Xp := (-B - sqrt(Delta))/(2*A);
          Yp := (-a/b*Xp-c/b);
        elseif(n=2):
          Xp := (-B + sqrt(Delta))/(2*A);
          Yp := (-a/b*Xp-c/b);
        fi
      elseif(Delta=0):
        Xp := -B/(2*A);
        Yp := -a/b*Xp-c/b;
      else:
        errmessage("! Intersection between line and circle does not exist")
      fi
    else: % sinon
      A := 1+(b/a)**2;
      B := 2*((b/a)*(c/a+Xc)-Yc);
      C := Yc**2+(c/a+Xc)**2-R**2;
      Delta := B**2-4*A*C;
      if(Delta>0):
        if(n=1):
          Yp := (-B - sqrt(Delta))/(2*A);
          Xp := (-b/a*Yp-c/a);
        elseif(n=2):
          Yp := (-B + sqrt(Delta))/(2*A);
          Xp := (-b/a*Yp-c/a);
        fi
      elseif(Delta=0):
        Yp := -B/(2*A);
        Xp := -b/a*Yp-c/a;
      else:
        errmessage("! Intersection between line and circle does not exist")
      fi
    fi
  fi
  Point(Xp,Yp)
enddef;

vardef LineCircleIntersection(expr D, CC, n) = IntersectionDroiteCercle(D, CC,
n) enddef;

% Inversion par rapport à un cercle -------------------------------------------
vardef Inversion (expr P, C) =
  save O,r,s,Cc,Pp,rr,Co,Po,ro,M,iM,R;
  O = Centre(C);
  r = Rayon(C);   
  if(gddT[P]="point"):
    s = r / Longueur(P,O);
    PairTOPoint(Pt(O) + (Pt(P) - Pt(O)) * s * s)
  elseif(gddT[P]="cercle"):
    Cc = Centre(P);
    rr = Rayon(P);
    if(Longueur(O,Cc)=rr): % si le cercle objet passe par le centre du cercle
      %                      % d’inversion
      % point du cercle sur rayon passant par les deux centre
      M = PairTOPoint(Pt(Cc)+Pt(Cc)-Pt(O));
      % inversion de ce point
      iM = Inversion(M,C);
      R = Droite(O,iM);
      DroitePerpendiculaire(R,iM)
    else:
      Pp = PairTOPoint(Pt(Cc)+rr*right);
      Co = Inversion(Cc,C);
      Po = Inversion(Pp,C);
      CercleCP(Co,Po)
    fi
  elseif(gddT[P]="droite"):
    R = DroitePerpendiculaire(P,O);
    iM = IntersectionDroites(R,P);
    M = Inversion(iM,C);
    Cc = Milieu(O,M);
    CercleCP(Cc,O)
  fi
enddef;

vardef Inverse(expr P,C) = Inversion(P,C) enddef;

% Axe radical entre deux cercles -----------------------------------------------
% drawing with metapost Thurston
vardef AxeRadical (expr CA, CB) =
  save O,ra,rb,d,t,C_CA,C_CB,_p,_q;
  pair _p,_q; 
  ra = Rayon(CA);
  rb = Rayon(CB);
  C_CA = Centre(CA);
  C_CB = Centre(CB);
  d = abs(Pt(C_CA) - Pt(C_CB));
  2t = 1 + (ra+rb) / d * (ra-rb) / d;
  _p := up scaled 2*gddU
  rotated angle (Pt(C_CB) - Pt(C_CA))
  shifted t[Pt(C_CA), Pt(C_CB)];
  _q := down scaled 2*gddU
  rotated angle (Pt(C_CB) - Pt(C_CA))
  shifted t[Pt(C_CA), Pt(C_CB)];
  Droite(_p,_q)
enddef;

vardef RadicalAxis(expr CA,CB) = AxeRadical(CA,CB) enddef;

% Centre radical de trois cercles ---------------------------------------------
% drawing with metapost Thurston
vardef CentreRadical (expr CA, CB, CC) =
  save DAB,DAC;
  DAB = AxeRadical(CA,CB);
  DAC = AxeRadical(CA,CC);
  IntersectionDroites(DAB,DAC)
enddef;

vardef RadicalCenter(expr CA,CB,CC) = CentreRadical(CA,CB,CC) enddef;

vardef Centre(expr c) = 
  % pour cercle, ellipse et polygone
  if((gddT[c] = "ellipse") or (gddT[c] = "cercle") or (gddT[c] = "polygone") or (gddT[c] = "hyperbole")): 
    gddA[c]
  fi
enddef;

vardef Center(expr c) = Centre(c) enddef;

% Chemin -----------------------------------------------------------------------
vardef Chemin (expr p) =
    gddT[incr gddO] = "chemin"; gddP[gddO] = p; gddO
enddef;

vardef Path(expr p) = Chemin(p) enddef;

vardef LigneBrisee(text t)=
  save n, _p,a,i;
  path _p;
  i:=0;
  for a=t:
    if(i=0):
      _p:=PairImp(a);
    else:
      _p:=_p--PairImp(a);  
    fi
    i:=i+1;
  endfor;
  Chemin(_p)
enddef;

vardef Polyline(text t) = LigneBrisee(t) enddef;

% Courbe -----------------------------------------------------------------------
vardef CourbeDat (expr s, c) =
    % c = 1 ou 0, selon que la courbe doit être fermée ou pas !
    gddT[incr gddO] = "courbe"; gddS[gddO] = s; gddA[gddO] = c; gddO
enddef;

vardef CurveData(expr s,c) = CourbeDat(s,c) enddef;

% Homothétie ------------------------------------------------------------------
vardef HomothetiePoint (expr p,o,k) = 
  % p : point
  % o : centre homothetie
  % k : facteur
  save OP,OM;
  pair OP,OM;
  OP = PairImp(p)-PairImp(o);
  OM = k*OP;
  PairTOPoint(PairImp(o)+OM)
enddef;

vardef HomothetyPoint(expr p,o,k) = HomothetiePoint(p,o,k) enddef;

vardef Homothetie (expr p,o,k) =
  % p : objet
  % o : centre homothetie
  % k : facteur
  save A,B,C,R,_p,X,n;
  numeric X[];
  pair _p;
  if((gddT[p]="point")):
    HomothetiePoint(p,o,k)
  elseif((gddT[p]="vecteur")):
    A = HomothetiePoint(p,o,k);
    Vecteur(gddA[A],gddB[A])
  elseif((gddT[p]="segment")):
    A = HomothetiePoint(gddA[p],o,k);
    B = HomothetiePoint(gddB[p],o,k);
    Segment(A,B)
  elseif((gddT[p]="droite")):
    A = HomothetiePoint(gddA[p],o,k);
    B = HomothetiePoint(gddB[p],o,k);
    Droite(A,B)
  elseif(gddT[p]="triangle"):
    A = HomothetiePoint(gddA[p],o,k);
    B = HomothetiePoint(gddB[p],o,k);
    C = HomothetiePoint(gddC[p],o,k);
    Triangle(A,B,C)
  elseif(gddT[p]="cercle"):
    A = HomothetiePoint(gddA[p],o,k);
    R = k*Rayon(p);
    Cercle(A,R)
  elseif(gddT[p]="ellipse"):
    % homothetie des foyers
    A = HomothetiePoint(gddD[p],o,k);
    B = HomothetiePoint(gddE[p],o,k);
    % echelle du demi grand axe
    R = k*gddX[p][1];
    EllipseF(A,B,R)
  elseif(gddT[p]="parabole"):
    A = HomothetiePoint(Foyer(p,1),o,k);
    B = Directrice(p,1);
    C = Homothetie(B,o,k);
    ParaboleFD(A,C)
  elseif(gddT[p]="hyperbole"):
    A = HomothetiePoint(Foyer(p,1),o,k);
    B = Directrice(p,1);
    C = Homothetie(B,o,k);
    HyperboleFD(A,C)
  elseif(gddT[p]="polygone"):
    % reprise du constructeur
    n = incr gddO; gddT[n] = "polygone";
    for i=1 upto gddB[p]:
      gddX[n][i] := HomothetiePoint(gddX[p][i],o,k);
    endfor
    gddB[n] := gddB[p]; % nombre de côté
    gddA[n] := HomothetiePoint(gddA[p],o,k); % centre
    n
  elseif(gddT[p]="chemin"):
    _p := PairImp(o);
    Chemin(((gddP[p] shifted (-_p)) scaled k) shifted _p) 
  fi
enddef;

vardef Homothety(expr p,o,k) = Homothetie(p,o,k) enddef;

% Symétrie ------------------------------------------------------------------
vardef SymetrieAxialePoint (expr p,d) = 
  % p : point
  % d : axe de symétrie
  save O,l,D;
  O = ProjectionPointSurDroite(p,d);
  l = Longueur(p,O);
  D = Droite(p,O);
  ReportSurDroite(p,D,2*l)
enddef;

vardef AxialSymmetryPoint(expr p,d) = SymetrieAxialePoint(p,d) enddef;

vardef SymetrieAxiale (expr p,d) =
  % p : objet
  % d : axe de symétrie
  save A,B,C,R,_p,X,n;
  numeric X[];
  pair _p;
  if((gddT[p]="point")):
    SymetrieAxialePoint(p,d)
  elseif((gddT[p]="vecteur")):
    A = SymetrieAxialePoint(p,d);
    Vecteur(gddA[A],gddB[A])
  elseif((gddT[p]="segment")):
    A = SymetrieAxialePoint(gddA[p],d);
    B = SymetrieAxialePoint(gddB[p],d);
    Segment(A,B)
  elseif((gddT[p]="droite")):
    A = SymetrieAxialePoint(gddA[p],d);
    B = SymetrieAxialePoint(gddB[p],d);
    Droite(A,B)
  elseif(gddT[p]="triangle"):
    A = SymetrieAxialePoint(gddA[p],d);
    B = SymetrieAxialePoint(gddB[p],d);
    C = SymetrieAxialePoint(gddC[p],d);
    Triangle(A,B,C)
  elseif(gddT[p]="cercle"):
    A = SymetrieAxialePoint(gddA[p],d);
    R = Rayon(p);
    Cercle(A,R)
  elseif(gddT[p]="ellipse"):
    % homothetie des foyers
    A = SymetrieAxialePoint(gddD[p],d);
    B = SymetrieAxialePoint(gddE[p],d);
    % echelle du demi grand axe
    R = gddX[p][1];
    EllipseF(A,B,R)
  elseif(gddT[p]="parabole"):
    A = SymetrieAxialePoint(Foyer(p,1),d);
    B = Directrice(p,1);
    C = SymetrieAxiale(B,d);
    ParaboleFD(A,C)
  elseif(gddT[p]="hyperbole"):
    A = SymetrieAxialePoint(Foyer(p,1),d);
    B = Directrice(p,1);
    C = SymetrieAxiale(B,d);
    HyperboleFD(A,C)
  elseif(gddT[p]="polygone"):
    % reprise du constructeur
    n = incr gddO; gddT[n] = "polygone";
    for i=1 upto gddB[p]:
      gddX[n][i] := SymetrieAxialePoint(gddX[p][i],d);
    endfor
    gddB[n] := gddB[p]; % nombre de côté
    gddA[n] := SymetrieAxialePoint(gddA[p],d); % centre
    n
  fi
enddef;

vardef AxialSymmetry(expr p,d) = SymetrieAxiale(p,d) enddef;

% Symétrie ------------------------------------------------------------------
vardef SymetrieCentralePoint (expr p,d) = 
  % p : point
  % d : point centre de symétrie
  save O,l,D;
  RotationCentre(p,d,Pi)
enddef;

vardef CentralSymmetryPoint(expr p,d) = SymetrieCentralePoint(p,d) enddef;

vardef SymetrieCentrale (expr p,d) =
  % p : objet
  % d : point centre de symétrie
  save A,B,C,R,X,n;
  numeric X[];
  if((gddT[p]="point")):
    SymetrieCentralePoint(p,d)
  elseif((gddT[p]="vecteur")):
    A = SymetrieCentralePoint(p,d);
    Vecteur(gddA[A],gddB[A])
  elseif((gddT[p]="segment")):
    A = SymetrieCentralePoint(gddA[p],d);
    B = SymetrieCentralePoint(gddB[p],d);
    Segment(A,B)
  elseif((gddT[p]="droite")):
    A = SymetrieCentralePoint(gddA[p],d);
    B = SymetrieCentralePoint(gddB[p],d);
    Droite(A,B)
  elseif(gddT[p]="triangle"):
    A = SymetrieCentralePoint(gddA[p],d);
    B = SymetrieCentralePoint(gddB[p],d);
    C = SymetrieCentralePoint(gddC[p],d);
    Triangle(A,B,C)
  elseif(gddT[p]="cercle"):
    A = SymetrieCentralePoint(gddA[p],d);
    R = Rayon(p);
    Cercle(A,R)
  elseif(gddT[p]="ellipse"):
    % homothetie des foyers
    A = SymetrieCentralePoint(gddD[p],d);
    B = SymetrieCentralePoint(gddE[p],d);
    % echelle du demi grand axe
    R = gddX[p][1];
    EllipseF(A,B,R)
  elseif(gddT[p]="parabole"):
    A = SymetrieCentralePoint(Foyer(p,1),d);
    B = Directrice(p,1);
    C = SymetrieCentrale(B,d);
    ParaboleFD(A,C)
  elseif(gddT[p]="hyperbole"):
    A = SymetrieCentralePoint(Foyer(p,1),d);
    B = Directrice(p,1);
    C = SymetrieCentrale(B,d);
    HyperboleFD(A,C)
  elseif(gddT[p]="polygone"):
    % reprise du constructeur
    n = incr gddO; gddT[n] = "polygone";
    for i=1 upto gddB[p]:
      gddX[n][i] := SymetrieCentralePoint(gddX[p][i],d);
    endfor
    gddB[n] := gddB[p]; % nombre de côté
    gddA[n] := SymetrieCentralePoint(gddA[p],d); % centre
    n
  fi
enddef;

vardef CentralSymmetry(expr p,d) = SymetrieCentrale(p,d) enddef;

%%% Similitude directe

vardef SimilitudeACentrePoint(expr p, d, t, k)=
  % p : point
  % d : point centre de similitude
  % t : angle de la rotation
  % k : paramètre de l’homothétie
  save A,B,C,D;
  pair A,B,C,D;
  A = PairImp(p);
  B = PairImp(d);
  C := (xpart B-xpart A,ypart B-ypart A);
  D := k*((xpart C)*cos(t)+(ypart C)*sin(t),-(xpart
  C)*sin(t)+(ypart C)*cos(t));
  Point(xpart A + xpart D,ypart A+ypart D)
enddef;

vardef CenterSimilarityPoint(expr p,d ,t, k) = SimilitudeACentrePoint(p,d, t, k) enddef;


vardef SimilitudeACentre (expr p,d,t, k) =
  % p : objet
  % d : point centre de similitude
  % t : angle de la rotation
  % k : paramètre de l’homothétie
  save A,B,C,R,X,n;
  numeric X[];
  if((gddT[p]="point")):
    SimilitudeACentrePoint(p,d,t, k)
  elseif((gddT[p]="vecteur")):
    A = SimilitudeACentrePoint(p,d,t, k);
    Vecteur(gddA[A],gddB[A])
  elseif((gddT[p]="segment")):
    A = SimilitudeACentrePoint(gddA[p],d,t, k);
    B = SimilitudeACentrePoint(gddB[p],d,t, k);
    Segment(A,B)
  elseif((gddT[p]="droite")):
    A = SimilitudeACentrePoint(gddA[p],d,t, k);
    B = SimilitudeACentrePoint(gddB[p],d,t, k);
    Droite(A,B)
  elseif(gddT[p]="triangle"):
    A = SimilitudeACentrePoint(gddA[p],d,t, k);
    B = SimilitudeACentrePoint(gddB[p],d,t, k);
    C = SimilitudeACentrePoint(gddC[p],d,t, k);
    Triangle(A,B,C)
  elseif(gddT[p]="cercle"):
    A = SimilitudeACentrePoint(gddA[p],d,t, k);
    R = Rayon(p);
    Cercle(A,R)
  elseif(gddT[p]="ellipse"):
    % similitude des foyers
    A = SimilitudeACentrePoint(gddD[p],d,t, k);
    B = SimilitudeACentrePoint(gddE[p],d,t, k);
    % echelle du demi grand axe
    R = gddX[p][1];
    EllipseF(A,B,R)
  elseif(gddT[p]="parabole"):
    A = SimilitudeACentrePoint(Foyer(p,1),d,t, k);
    B = Directrice(p,1);
    C = SimilitudeACentre(B,d,t, k);
    ParaboleFD(A,C)
  elseif(gddT[p]="hyperbole"):
    A = SimilitudeACentrePoint(Foyer(p,1),d,t, k);
    B = Directrice(p,1);
    C = SimilitudeACentre(B,d,t, k);
    HyperboleFD(A,C)
  elseif(gddT[p]="polygone"):
    % reprise du constructeur
    n = incr gddO; gddT[n] = "polygone";
    for i=1 upto gddB[p]:
      gddX[n][i] := SimilitudeACentrePoint(gddX[p][i],d,t,k);
    endfor
    gddB[n] := gddB[p]; % nombre de côté
    gddA[n] := SimilitudeACentrePoint(gddA[p],d,t, k); % centre
    n
  fi
enddef;

vardef CenterSimilarity(expr p,d ,t, k) = SimilitudeACentre(p,d, t, k) enddef;

vardef PointDe(expr o,t)=
  % t: paramètre le long de l’objet entre 0 et 1
  % o: objet mpgdd
  save _l,_t,_p;
  path _p;
  if((gddT[o]="point")):
    o 
  else:
    _p := gddTraceObjet(o);
    _l := length(_p);
    PointImp(point (t*_l) of _p)
  fi
enddef;

vardef PointOf(expr o,t) = PointDe(o,t) enddef;

%%% ----------------------------------------------------------------------------
%%% Définitions liées au repère
%%% ----------------------------------------------------------------------------

% Définition de l'origine ------------------------------------------------------
origine = Point(0,0);

% Définition des axes ----------------------------------------------------------
AxeOx = Droite(origin,right);
AxeOy = Droite(origin,up);

Xaxis = AxeOx;
Yaxis = AxeOy;

%%% ----------------------------------------------------------------------------
%%% Calculs MetaPost
%%% ----------------------------------------------------------------------------

vardef mpIntersectionDroites (expr a,b,c,d) =
    save t; pair t; t = whatever [a,b]; t = whatever [c,d]; t
enddef;

%%% --------------------------------------------------------------------------
%%% Tracés des courbes
%%% --------------------------------------------------------------------------
vardef gddTraceCourbe (expr f, c) =
    save s; gdata(f, s, if i>1:..fi (scantokens s1,scantokens s2)) if c>0:..cycle fi
enddef;

%%% ----------------------------------------------------------------------------
%%% Tracés des objets
%%% ----------------------------------------------------------------------------

gddExtensionDroite := 10;

def ParametreExtensionDroite(expr t) =
  gddExtensionDroite := t
enddef;

def SetLineExtension(expr t) =
  gddExtensionDroite := t
enddef;

def gddTracePolygone expr o=
%show "test";
  for i=1 upto gddB[o]:
    PointTOPair(gddX[o][i])--
  endfor
  cycle
enddef;

def gddTraceEllipse expr o=
  (fullcircle xscaled (2*gddX[o][1]) yscaled (2*gddX[o][2])) rotated gddX[o][4] shifted PairImp(gddA[o])
enddef;

def gddTraceObjet expr o =
  if gddT[o] = "triangle":
    PointTOPair(gddA[o])--PointTOPair(gddB[o])--
    PointTOPair(gddC[o])--cycle
  elseif (gddT[o] = "segment") :
    PointTOPair(gddA[o])--PointTOPair(gddB[o])
  elseif (gddT[o] = "vecteur"):
    (0,0)--(gddA[o],gddB[o])
  elseif gddT[o] = "droite":
    (gddExtensionDroite [ PointTOPair(gddA[o]), PointTOPair(gddB[o]) ]) --
    (gddExtensionDroite [ PointTOPair(gddB[o]), PointTOPair(gddA[o]) ])
  elseif gddT[o] = "cercle":
    fullcircle scaled (gddB[o]*2) shifted PointTOPair(gddA[o])
  elseif gddT[o] = "ellipse":
    gddTraceEllipse (o)
  elseif ((gddT[o] = "chemin") or (gddT[o] = "parabole")):
    gddP[o]
  elseif gddT[o] = "courbe":
    %message "This is a debug message.";
    gddTraceCourbe (gddS[o] & ".dat", gddA[o])
  elseif gddT[o] = "polygone":
    gddTracePolygone (o)  
  fi
enddef;

vardef gddObjectDrawing(expr o) = gddTraceObjet(o) enddef;

def trace expr p =
  if path p or picture p or pair p: draw (p)
  elseif gddT[p] = "vecteur": drawarrow (gddTraceObjet p)
  else: draw (gddTraceObjet p)
  fi
  gddEnPlace
enddef;

let gddDraw = trace;

def fleche expr p =
  if path p or picture p: drawarrow (p)
  else: drawarrow (gddTraceObjet p)
  fi gddEnPlace
enddef;

let gddArrow = fleche;

def colorie expr p =
  if path p: fill p else: fill (gddTraceObjet p) fi gddEnPlace
enddef;

let gddFill = colorie;

% inspiration geometriesyr16

% _hatching parameter
numeric _ExtHatch,_ExtNombre;
_ExtHatch := 40;
def ExtensionHachure(expr t)=
  _ExtHatch:=t;
enddef;

let HatchExtension = ExtensionHachure;

_ExtNombre := 200;

def _hachure_pattern = 
  on0.2 off0.2
enddef;

def SchemaHachure(text t) =  def _hachure_pattern = t enddef enddef;

let HatchPattern = SchemaHachure;

vardef hachure(expr p,a,e,t) =
  save $,support;
  picture $;
  path support;
  % le trait de base
  support=((-_ExtHatch,0)--(_ExtHatch,0)) rotated a;
  if t=1:
    drawoptions(dashed evenly scaled (1.0/(gddU)));
  elseif t=2:
    drawoptions(dashed dashpattern(on12bp off6bp
    on3bp off6bp) scaled (1.0/(gddU)));
  elseif t=3:
    drawoptions(dashed withdots scaled (1.0/(gddU)));
  elseif t=4:
    drawoptions(dashed dashpattern(_hachure_pattern));  
  fi;
  $ = image(
    for j=-_ExtNombre upto _ExtNombre:
      % on test s’il y a intersection (pour ne pas dessiner des choses inutiles)
      if ((support shifted (e*j*(sind(a),-cosd(a)) if(t>0): +e*(j mod
	      3-1)*(cosd(a),sind(a)) fi) intersectiontimes (gddTraceObjet(p)))<>(-1,-1)):
	      draw support shifted (e*j*(sind(a),-cosd(a)) if(t>0): +e*(j mod
	      3-1)*(cosd(a),sind(a)) fi) withpen (pencircle scaled (1.0/(2gddU)));
      fi
    endfor;
    );
  clip $ to gddTraceObjet(p);
  drawoptions();
  $
enddef;

vardef gddHatch(expr p,a,e,t) = hachure(p,a,e,t) enddef;

def fermeture expr p =
  if path p: p else: (gddTraceObjet p) fi -- cycle
enddef;

let gddClose = fermeture;

def avecCrayon(expr t,c) = withpen pencircle scaled t withcolor c enddef;

let withPen = avecCrayon;

%%% --------------------------------------------------------------------------
%%% Marque et label des points
%%% --------------------------------------------------------------------------
gddTaillePoint  := 3;                % Diamètre des cercles marquant un point.

def ParametreTaillePoint(expr t) =
  gddTaillePoint := t
enddef;
let SetPointSize = ParametreTaillePoint;

color gddCouleurPoint;               % Couleur de fond de la marque d'un point.
gddCouleurPoint := white;

def ParametreCouleurPoint(expr c) =
  gddCouleurPoint := c
enddef;
let SetPointColor = ParametreCouleurPoint;

gddTailleLabel  := 1;              % Taille d'un label.

def ParametreTailleLabel(expr t) =
  gddTailleLabel := t
enddef;
let SetLabelSize = ParametreTailleLabel;

string gddPointeType;
gddPointeType := "pointe";

def ParametrePointeType(expr t) =
  gddPointeType := t
enddef;
let SetPointType = ParametrePointeType;


def gddPointe(expr p) =
  fill (fullcircle scaled gddTaillePoint) shifted (p gddEnPlace)
    withcolor gddCouleurPoint;
  draw (fullcircle scaled gddTaillePoint) shifted (p gddEnPlace)
enddef;

def gddDisque(expr p) =
  fill (fullcircle scaled gddTaillePoint) shifted (p gddEnPlace)
enddef;

def gddCroix(expr p) =
  draw (gdd_marque2 scaled (gddTaillePoint/7)) shifted (p gddEnPlace)
enddef;

def pointe expr p =
  if(gddPointeType="pointe"):
    gddPointe(if pair p: p else: PointTOPair(p) fi)
  elseif(gddPointeType="croix"):
    gddCroix(if pair p: p else: PointTOPair(p) fi)
  elseif(gddPointeType="disque"):
    gddDisque(if pair p: p else: PointTOPair(p) fi)
  fi
enddef;

def pointes text t = 
  for p=t:
    pointe p;
  endfor;
enddef;

let gddDrawPoint = pointe;

let gddDrawPoints = pointes;



%%% --------------------------------------------------------------------------
%%% Signes
%%% --------------------------------------------------------------------------
def SigneOrtho(expr a,b,c,x) =
  (Pt(b) + x * unitvector(Pt(a)-Pt(b)))
   -- (Pt(b) + x * unitvector(Pt(a) - Pt(b))
   + x * unitvector(Pt(c) - Pt(b)))
   -- (Pt(b) + x * unitvector(Pt(c) - Pt(b)))
enddef;

let OrthoSign = SigneOrtho;

picture gdd_marque[];
gdd_marque1 = image(draw (-7,-7)--(7,7));
gdd_marque2 = image(draw (-7,-7)--(7,7); draw (-7,7)--(7,-7));
gdd_marque3 = image(draw (-9,-7)--(5,7); draw (-5,-7)--(9,7);
  draw (-7,7)--(7,-7));
gdd_marque4 = image(draw (-9,-7)--(5,7); draw (-5,-7)--(9,7);
  draw (-9,7)--(5,-7); draw (-5,7)--(9,-7));

def Marque(expr a,b,n) =
  gdd_marque[n] rotated angle(PairImp(b)-PairImp(a)) scaled (0.5/gddU)
   shifted 0.5[PairImp(a),PairImp(b)]
enddef;

vardef Mark(expr a,b,n) = Marque(a,b,n) enddef;

vardef MarqueTrait(expr s,n) =
  save chemin;
  path chemin;
  if path s:
    chemin:=s;
  elseif(numeric s):
    chemin:= gddTraceObjet(s)
  fi;
  gdd_marque[n] rotated angle(point 0 of chemin-point length(chemin) of chemin) scaled (0.5/gddU)
   shifted _milieuChemin(chemin,0.5)
enddef;

vardef LineMark(expr s,n) = MarqueTrait(s,n) enddef;

%%% --------------------------------------------------------------------------
%%% Fenêtrage
%%% --------------------------------------------------------------------------

def Fenetre(expr xg,yg,xd,yd) =
  gddXG := xg;
  gddYB := yg;
  gddXD := xd;
  gddYH := yd;
  extra_endfig := "gddFenetre;" & extra_endfig;
enddef;

let Window = Fenetre;

def FinFenetre = 
  extra_endfig := "";
enddef;

let EndWindow = FinFenetre;

def gddFenetre =
  clip currentpicture to
   ((gddXG,gddYB)--(gddXG,gddYH)--(gddXD,gddYH)--(gddXD,gddYB)--cycle)
   gddEnPlace;
  draw ((gddXG,gddYB)--(gddXG,gddYH)--(gddXD,gddYH)--(gddXD,gddYB)--cycle)
   gddEnPlace withpen pencircle scaled 1 withcolor background;
enddef;

def gddClip(expr p) =
  clip currentpicture to (gddTraceObjet(p)) gddEnPlace;
enddef;

endinput
