#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \repeat unfold 8 hh8
    }
    \\ {
        bd4 sn8 bd8 r4 sn4
    }>>
}