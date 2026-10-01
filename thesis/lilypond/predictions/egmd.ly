#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}
\paper {
    line-width = 150\mm 
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \repeat unfold 13 hh8
    }
    \\ {
        bd16 bd16 s8 sn4 r16 sn16 \acciaccatura sn64(bd8) sn8 bd8 | \acciaccatura bd32(bd16) bd16 s8 sn8 r16 sn32 sn32 r16 sn16
    }>>
}