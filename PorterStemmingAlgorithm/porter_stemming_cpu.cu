#include <string.h>  /* for memmove */

#include "porter_stemming_cpu.cuh"

void toLowerCPU(char *p) 
{
switch(*p)
{
  case 'A':*p='a'; return;
  case 'B':*p='b'; return;
  case 'C':*p='c'; return;
  case 'D':*p='d'; return;
  case 'E':*p='e'; return;
  case 'F':*p='f'; return;
  case 'G':*p='g'; return;
  case 'H':*p='h'; return;
  case 'I':*p='i'; return;
  case 'J':*p='j'; return;
  case 'K':*p='k'; return;
  case 'L':*p='l'; return;
  case 'M':*p='m'; return;
  case 'N':*p='n'; return;
  case 'O':*p='o'; return;
  case 'P':*p='p'; return;
  case 'Q':*p='q'; return;
  case 'R':*p='r'; return;
  case 'S':*p='s'; return;
  case 'T':*p='t'; return;
  case 'U':*p='u'; return;
  case 'V':*p='v'; return;
  case 'W':*p='w'; return;
  case 'X':*p='x'; return;
  case 'Y':*p='y'; return;
  case 'Z':*p='z'; return;
};
return ;
}
void toUpperCPU(char *p) 
{
switch(*p)
{
  case 'a':*p='A'; return;
  case 'b':*p='B'; return;
  case 'c':*p='C'; return;
  case 'd':*p='D'; return;
  case 'e':*p='E'; return;
  case 'f':*p='F'; return;
  case 'g':*p='G'; return;
  case 'h':*p='H'; return;
  case 'i':*p='I'; return;
  case 'j':*p='J'; return;
  case 'k':*p='K'; return;
  case 'l':*p='L'; return;
  case 'm':*p='M'; return;
  case 'n':*p='N'; return;
  case 'o':*p='O'; return;
  case 'p':*p='P'; return;
  case 'q':*p='Q'; return;
  case 'r':*p='R'; return;
  case 's':*p='S'; return;
  case 't':*p='T'; return;
  case 'u':*p='U'; return;
  case 'v':*p='V'; return;
  case 'w':*p='W'; return;
  case 'x':*p='X'; return;
  case 'y':*p='Y'; return;
  case 'z':*p='Z'; return;
};
return ;
}



/* The main part of the stemming algorithm starts here. b is a buffer

 holding a word to be stemmed. The letters are in data_[k0], data_[k0+1] ...

 ending at data_[k]. In fact k0 = 0 in this demo program. k is readjusted

 downwards as the stemming progresses. Zero termination is not in fact

 used in the algorithm.

 

 Note that only lower case sequences are stemmed. Forcing to lower case

 should be done before stem(...) is called.

 */





bool PorterStemmingCPU::cons(int i)

{  switch (data_[i])

    {  case 'a': case 'e': case 'i': case 'o': case 'u': return false;

        case 'y': return (i==k0) ? true : !cons(i-1);

        default: return true;

    }

}



int PorterStemmingCPU::m()

{

    int n = 0;

    int i = k0;

    while(true)

        {  if (i > j) return n;

            if (! cons(i)) break; i++;

        }

    i++;

    while(true)

        {  while(true)

            {  if (i > j) return n;

                if (cons(i)) break;

                i++;

            }

            i++;

            n++;

            while(true)

                {  if (i > j) return n;

                    if (! cons(i)) break;

                    i++;

                }

            i++;

        }

}



int PorterStemmingCPU::vowelinstem()

{  int i; for (i = k0; i <= j; i++) if (! cons(i)) return true;

    return false;

}



int PorterStemmingCPU::doublec(int j)

{  if (j < k0+1) return false;

    if (data_[j] != data_[j-1]) return false;

    return cons(j);

}



int PorterStemmingCPU::cvc(int i)

{  if (i < k0+2 || !cons(i) || cons(i-1) || !cons(i-2)) return false;

    {  int ch = data_[i];

        if (ch == 'w' || ch == 'x' || ch == 'y') return false;

    }

    return true;

}





int PorterStemmingCPU::ends(const char * s)

{  int length = s[0];

    if (s[length] != data_[k]) return false; /* tiny speed-up */

    if (length > k-k0+1) return false;

    for (int i = 0; i < length; ++i) {

        if (data_[k-length+1+i] != s[1+i])

            return false;

    }

//    if (memcmp(data_+k-length+1,s+1,length) != 0) return false;

    j = k-length;

    return true;

}



void PorterStemmingCPU::setto(const char * s)

{  int length = s[0];

    for (int i = 0; i < length; ++i) {

        data_[i+j+1] = s[i+1];

    }

//    memmove(data_+j+1,s+1,length);

    k = j+length;

}



void PorterStemmingCPU::r(const char * s) { if (m() > 0) setto(s); }



void PorterStemmingCPU::step1ab()

{  if (data_[k] == 's')

    {  if (ends("\04" "sses")) k -= 2; else

        if (ends("\03" "ies")) setto("\01" "i"); else

            if (data_[k-1] != 's') k--;

    }

    if (ends("\03" "eed")) { if (m() > 0) k--; } else

        if ((ends("\02" "ed") || ends("\03" "ing")) && vowelinstem())

            {  k = j;

                if (ends("\02" "at")) setto("\03" "ate"); else

                    if (ends("\02" "bl")) setto("\03" "ble"); else

                        if (ends("\02" "iz")) setto("\03" "ize"); else

                            if (doublec(k))

                                {  k--;

                                    {  int ch = data_[k];

                                        if (ch == 'l' || ch == 's' || ch == 'z') k++;

                                    }

                                }

                            else if (m() == 1 && cvc(k)) setto("\01" "e");

            }

}



void PorterStemmingCPU::step1c() {

    if (ends("\01" "y") && vowelinstem())

        data_[k] = 'i';

}



void PorterStemmingCPU::step2() { switch (data_[k-1])

    {

        case 'a': if (ends("\07" "ational")) { r("\03" "ate"); break; }

        if (ends("\06" "tional")) { r("\04" "tion"); break; }

        break;

        case 'c': if (ends("\04" "enci")) { r("\04" "ence"); break; }

        if (ends("\04" "anci")) { r("\04" "ance"); break; }

        break;

        case 'e': if (ends("\04" "izer")) { r("\03" "ize"); break; }

        break;

        case 'l': if (ends("\03" "bli")) { r("\03" "ble"); break; } /*-DEPARTURE-*/

        

        /* To match the published algorithm, replace this line with

         case 'l': if (ends("\04" "abli")) { r("\04" "able"); break; } */

        

        if (ends("\04" "alli")) { r("\02" "al"); break; }

        if (ends("\05" "entli")) { r("\03" "ent"); break; }

        if (ends("\03" "eli")) { r("\01" "e"); break; }

        if (ends("\05" "ousli")) { r("\03" "ous"); break; }

        break;

        case 'o': if (ends("\07" "ization")) { r("\03" "ize"); break; }

        if (ends("\05" "ation")) { r("\03" "ate"); break; }

        if (ends("\04" "ator")) { r("\03" "ate"); break; }

        break;

        case 's': if (ends("\05" "alism")) { r("\02" "al"); break; }

        if (ends("\07" "iveness")) { r("\03" "ive"); break; }

        if (ends("\07" "fulness")) { r("\03" "ful"); break; }

        if (ends("\07" "ousness")) { r("\03" "ous"); break; }

        break;

        case 't': if (ends("\05" "aliti")) { r("\02" "al"); break; }

        if (ends("\05" "iviti")) { r("\03" "ive"); break; }

        if (ends("\06" "biliti")) { r("\03" "ble"); break; }

        break;

        case 'g': if (ends("\04" "logi")) { r("\03" "log"); break; } /*-DEPARTURE-*/

        

        /* To match the published algorithm, delete this line */

        

    }

}



void PorterStemmingCPU::step3() { switch (data_[k])

    {

        case 'e': if (ends("\05" "icate")) { r("\02" "ic"); break; }

        if (ends("\05" "ative")) { r("\00" ""); break; }

        if (ends("\05" "alize")) { r("\02" "al"); break; }

        break;

        case 'i': if (ends("\05" "iciti")) { r("\02" "ic"); break; }

        break;

        case 'l': if (ends("\04" "ical")) { r("\02" "ic"); break; }

        if (ends("\03" "ful")) { r("\00" ""); break; }

        break;

        case 's': if (ends("\04" "ness")) { r("\00" ""); break; }

        break;

    } }



/* step4() takes off -ant, -ence etc., in context <c>vcvc<v>. */



void PorterStemmingCPU::step4()

{  switch (data_[k-1])

    {  case 'a': if (ends("\02" "al")) break; return;

        case 'c': if (ends("\04" "ance")) break;

            if (ends("\04" "ence")) break; return;

        case 'e': if (ends("\02" "er")) break; return;

        case 'i': if (ends("\02" "ic")) break; return;

        case 'l': if (ends("\04" "able")) break;

            if (ends("\04" "ible")) break; return;

        case 'n': if (ends("\03" "ant")) break;

            if (ends("\05" "ement")) break;

            if (ends("\04" "ment")) break;

            if (ends("\03" "ent")) break; return;

        case 'o': if (ends("\03" "ion") && j >= k0 && (data_[j] == 's' || data_[j] == 't')) break;

            if (ends("\02" "ou")) break; return;

            /* takes care of -ous */

        case 's': if (ends("\03" "ism")) break; return;

        case 't': if (ends("\03" "ate")) break;

            if (ends("\03" "iti")) break; return;

        case 'u': if (ends("\03" "ous")) break; return;

        case 'v': if (ends("\03" "ive")) break; return;

        case 'z': if (ends("\03" "ize")) break; return;

        default: return;

    }

    if (m() > 1) k = j;

}



/* step5() removes a final -e if m() > 1, and changes -ll to -l if

 m() > 1. */



void PorterStemmingCPU::step5()

{  j = k;

    if (data_[k] == 'e')

        {  int a = m();

            if (a > 1 || (a == 1 && !cvc(k-1))) k--;

        }

    if (data_[k] == 'l' && doublec(k) && m() > 1) k--;

}





int PorterStemmingCPU::stem(int start, int end)

{   k = end; k0 = start; /* copy the parameters into statics */



    if (k <= k0+1) return k; /*-DEPARTURE-*/

//    data_[start] = '>';

//    data_[end] = '<';

    step1ab();

    if (k > k0) {

        step1c(); step2(); step3(); step4(); step5();

    }

    return k;

}