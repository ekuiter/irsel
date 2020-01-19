%------------------------------------------------------------------------------
% File     : NLP261+1 : TPTP v7.3.0. Bugfixed v4.0.1.
% Domain   : Commonsense Reasoning
% Problem  : Cytogeneticist is a hyponym of scientist
% Version  : Especial.
% English  :

% Refs     : [Fel98] Felbaum (1998), WordNet: An Electronic Lexical Databas
%          : [deM09] de Melo (2009), Email to Geoff Sutcliffe
% Source   : [deM09]
% Names    : wn2 [deM09]

% Status   : Theorem
% Rating   : 0.56 v7.3.0, 0.71 v7.2.0, 0.50 v7.0.0, 0.71 v6.4.0, 0.79 v6.3.0, 0.77 v6.2.0, 0.82 v6.1.0, 0.92 v6.0.0, 0.75 v5.5.0, 0.96 v5.3.0, 1.00 v5.2.0, 0.93 v5.0.0, 0.95 v4.1.0, 1.00 v4.0.1
% Syntax   : Number of formulae    : 1026861 (1026858 unit)
%            Number of atoms       : 1026865 (   0 equality)
%            Maximal formula depth :    6 (   1 average)
%            Number of connectives :    4 (   0   ~;   0   |;   1   &)
%                                         (   0 <=>;   3  =>;   0  <=)
%                                         (   0 <~>;   0  ~|;   0  ~&)
%            Number of predicates  :   30 (   0 propositional; 2-2 arity)
%            Number of functors    : 383180 (383180 constant; 0-0 arity)
%            Number of variables   :    7 (   0 sgn;   7   !;   0   ?)
%            Maximal term depth    :    1 (   1 average)
% SPC      : FOF_THM_RFO_NEQ

% Comments : n9986904 (cytogeneticist) is a hyponym of n10126424 (geneticist),
%            which is a hyponym of n9855630 (biologist), which is a hyponym of
%            n10560637 (scientist), which is a hyponym of n7846 (individual).
% Bugfixes : v4.0.1 - Added _c to constants that were the same as predicates.
%------------------------------------------------------------------------------
%----Include axioms from SUMO
include('Axioms/NLP001+0.ax').
%------------------------------------------------------------------------------
fof(axiom1,axiom,(
    ! [X,Y,Z] :
      ( ( hypernym(X,Y)
        & hypernym(Y,Z) )
     => hypernym(X,Z) ) )).

fof(axiom2,axiom,(
    ! [X,Y] :
      ( hypernym(X,Y)
     => hyponym(Y,X) ) )).

fof(axiom3,axiom,(
    ! [X,Y] :
      ( hyponym(X,Y)
     => hypernym(Y,X) ) )).

fof(hypernym_transitiviy_1,conjecture,(
    hypernym(n9986904,n10560637) )).

%------------------------------------------------------------------------------
