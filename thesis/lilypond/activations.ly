#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        cymc8 \repeat unfold 6 hh8 cymc8
    }
    \\ {
        bd4 sn8 bd8 bd8 r16 bd16 sn4
    }>>
}