class maths {
    static zerosX2X2(x) {
        const zs = [];
        while (x--) { zs.push([0, 0]); }
        return zs;
    }
    static mulItems(items, multiplier) {
        return [items[0] * multiplier, items[1] * multiplier];
    }
    static mulMatrix(m1, m2) {
        return (m1[0] * m2[0]) + (m1[1] * m2[1]);
    }
    static subtract(arr1, arr2) {
        return [arr1[0] - arr2[0], arr1[1] - arr2[1]];
    }
    static addArrays(arr1, arr2) {
        return [arr1[0] + arr2[0], arr1[1] + arr2[1]];
    }
    static addItems(items, addition) {
        return [items[0] + addition, items[1] + addition];
    }
    static sum(items) {
        return items.reduce((sum, x) => sum + x);
    }
    static dot(m1, m2) {
        return maths.mulMatrix(m1, m2);
    }
    static vectorLen(v) {
        const a = v[0];
        const b = v[1];
        return Math.sqrt((a * a) + (b * b));
    }
    static divItems(items, divisor) {
        return [items[0] / divisor, items[1] / divisor];
    }
    static squareItems(items) {
        const a = items[0];
        const b = items[1];
        return [a * a, b * b];
    }
    static normalize(v) {
        return this.divItems(v, this.vectorLen(v));
    }
}

class bezier {
    static q(ctrlPoly, t) {
        const tx = 1.0 - t;
        const pA = maths.mulItems(ctrlPoly[0], tx * tx * tx);
        const pB = maths.mulItems(ctrlPoly[1], 3 * tx * tx * t);
        const pC = maths.mulItems(ctrlPoly[2], 3 * tx * t * t);
        const pD = maths.mulItems(ctrlPoly[3], t * t * t);
        return maths.addArrays(maths.addArrays(pA, pB), maths.addArrays(pC, pD));
    }
    static qprime(ctrlPoly, t) {
        const tx = 1.0 - t;
        const pA = maths.mulItems(maths.subtract(ctrlPoly[1], ctrlPoly[0]), 3 * tx * tx);
        const pB = maths.mulItems(maths.subtract(ctrlPoly[2], ctrlPoly[1]), 6 * tx * t);
        const pC = maths.mulItems(maths.subtract(ctrlPoly[3], ctrlPoly[2]), 3 * t * t);
        return maths.addArrays(maths.addArrays(pA, pB), pC);
    }
    static qprimeprime(ctrlPoly, t) {
        return maths.addArrays(
            maths.mulItems(maths.addArrays(maths.subtract(
                ctrlPoly[2],
                maths.mulItems(ctrlPoly[1], 2),
            ), ctrlPoly[0]), 6 * (1.0 - t)),
            maths.mulItems(maths.addArrays(maths.subtract(
                ctrlPoly[3],
                maths.mulItems(ctrlPoly[2], 2),
            ), ctrlPoly[1]), 6 * t),
        );
    }
}

function chordLengthParameterize(points) {
    let u = [];
    let currU;
    let prevU;
    let prevP;

    points.forEach((p, i) => {
        currU = i ? prevU + maths.vectorLen(maths.subtract(p, prevP)) : 0;
        u.push(currU);

        prevU = currU;
        prevP = p;
    });
    u = u.map(x => x / prevU);
    return u;
}

function generateBezier(points, parameters, leftTangent, rightTangent) {
    let a;
    let i;
    let len;
    let tmp;
    let u;
    let ux;
    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];
    const bezCurve = [firstPoint, null, null, lastPoint];
    const A = maths.zerosX2X2(parameters.length);

    for (i = 0, len = parameters.length; i < len; i += 1) {
        u = parameters[i];
        ux = 1 - u;
        a = A[i];

        a[0] = maths.mulItems(leftTangent, (3 * u) * (ux * ux));
        a[1] = maths.mulItems(rightTangent, (3 * ux) * (u * u));
    }

    const C = [[0, 0], [0, 0]];
    const X = [0, 0];
    for (i = 0, len = points.length; i < len; i += 1) {
        u = parameters[i];
        a = A[i];

        C[0][0] += maths.dot(a[0], a[0]);
        C[0][1] += maths.dot(a[0], a[1]);
        C[1][0] += maths.dot(a[0], a[1]);
        C[1][1] += maths.dot(a[1], a[1]);

        tmp = maths.subtract(points[i], bezier.q([
            firstPoint, firstPoint, lastPoint, lastPoint,
        ], u));

        X[0] += maths.dot(a[0], tmp);
        X[1] += maths.dot(a[1], tmp);
    }
    const detC0C1 = (C[0][0] * C[1][1]) - (C[1][0] * C[0][1]);
    const detC0X = (C[0][0] * X[1]) - (C[1][0] * X[0]);
    const detXC1 = (X[0] * C[1][1]) - (X[1] * C[0][1]);
    const alphaL = detC0C1 === 0 ? 0 : detXC1 / detC0C1;
    const alphaR = detC0C1 === 0 ? 0 : detC0X / detC0C1;

    const segLength = maths.vectorLen(maths.subtract(firstPoint, lastPoint));
    const epsilon = 1.0e-6 * segLength;
    if (alphaL < epsilon || alphaR < epsilon) {
        bezCurve[1] = maths.addArrays(firstPoint, maths.mulItems(leftTangent, segLength / 3.0));
        bezCurve[2] = maths.addArrays(lastPoint, maths.mulItems(rightTangent, segLength / 3.0));
    } else {
        bezCurve[1] = maths.addArrays(firstPoint, maths.mulItems(leftTangent, alphaL));
        bezCurve[2] = maths.addArrays(lastPoint, maths.mulItems(rightTangent, alphaR));
    }
    return bezCurve;
}

function mapTtoRelativeDistances(bez, BParts) {
    let BTCurr;
    let BTDist = [0];
    let BTPrev = bez[0];
    let sumLen = 0;

    for (let i = 1; i <= BParts; i += 1) {
        BTCurr = bezier.q(bez, i / BParts);
        sumLen += maths.vectorLen(maths.subtract(BTCurr, BTPrev));
        BTDist.push(sumLen);
        BTPrev = BTCurr;
    }

    BTDist = BTDist.map(x => x / sumLen);
    return BTDist;
}

function findT(bez, param, tDistMap, BParts) {
    if (param < 0) {
        return 0;
    }
    if (param > 1) {
        return 1;
    }
    let lenMax;
    let lenMin;
    let tMax;
    let tMin;
    let t;
    for (let i = 1; i <= BParts; i += 1) {
        if (param <= tDistMap[i]) {
            tMin = (i - 1) / BParts;
            tMax = i / BParts;
            lenMin = tDistMap[i - 1];
            lenMax = tDistMap[i];
            t = (param - lenMin) / (((lenMax - lenMin) * (tMax - tMin)) + tMin);
            break;
        }
    }
    return t;
}

function computeMaxError(points, bez, parameters) {
    let i;
    let count;
    let point;
    let maxDist = 0;
    let splitPoint = points.length / 2;
    const tDistMap = mapTtoRelativeDistances(bez, 10);

    for (i = 0, count = points.length; i < count; i += 1) {
        point = points[i];
        const t = findT(bez, parameters[i], tDistMap, 10);
        const v = maths.subtract(bezier.q(bez, t), point);
        const dist = (v[0] * v[0]) + (v[1] * v[1]);
        if (dist > maxDist) {
            maxDist = dist;
            splitPoint = i;
        }
    }
    return [maxDist, splitPoint];
}

function generateAndReport(
    points, paramsOrig, paramsPrime,
    leftTangent, rightTangent, progressCallback,
) {
    const bezCurve =
    generateBezier(points, paramsPrime, leftTangent, rightTangent, progressCallback);
    const [maxError, splitPoint] = computeMaxError(points, bezCurve, paramsOrig);

    if (progressCallback) {
        progressCallback({
            bez: bezCurve,
            points,
            params: paramsOrig,
            maxErr: maxError,
            maxPoint: splitPoint,
        });
    }

    return [bezCurve, maxError, splitPoint];
}

function newtonRaphsonRootFind(bez, point, u) {
    const d = maths.subtract(bezier.q(bez, u), point);
    const qprime = bezier.qprime(bez, u);
    const numerator = maths.mulMatrix(d, qprime);
    const denominator = maths.sum(maths.squareItems(qprime)) + (2 * maths.mulMatrix(
        d,
        bezier.qprimeprime(bez, u),
    ));

    if (denominator === 0) {
        return u;
    }
    return u - (numerator / denominator);
}

function reparameterize(Bezier, points, parameters) {
    return parameters.map((p, i) => newtonRaphsonRootFind(Bezier, points[i], p));
}

function fitCubic(points, leftTangent, rightTangent, error, progressCallback) {
    const MaxIterations = 20;
    let bezCurve;
    let uPrime;
    let maxError;
    let prevErr;
    let splitPoint;
    let prevSplit;
    let centerVector;
    let beziers;
    let dist;
    let i;

    if (points.length === 2) {
        dist = maths.vectorLen(maths.subtract(points[0], points[1])) / 3.0;
        bezCurve = [
            points[0],
            maths.addArrays(points[0], maths.mulItems(leftTangent, dist)),
            maths.addArrays(points[1], maths.mulItems(rightTangent, dist)),
            points[1],
        ];
        return [bezCurve];
    }

    const u = chordLengthParameterize(points);
    [bezCurve, maxError, splitPoint] = generateAndReport(
        points, u, u, leftTangent,
        rightTangent, progressCallback,
    );

    if (maxError < error) {
        return [bezCurve];
    }
    if (maxError < (error * error)) {
        uPrime = u;
        prevErr = maxError;
        prevSplit = splitPoint;

        for (i = 0; i < MaxIterations; i += 1) {
            uPrime = reparameterize(bezCurve, points, uPrime);
            [bezCurve, maxError, splitPoint] = generateAndReport(
                points, u, uPrime, leftTangent,
                rightTangent, progressCallback,
            );

            if (maxError < error) {
                return [bezCurve];
            } else if (splitPoint === prevSplit) {
                const errChange = maxError / prevErr;
                if ((errChange > 0.9999) && (errChange < 1.0001)) {
                    break;
                }
            }
            prevErr = maxError;
            prevSplit = splitPoint;
        }
    }
    beziers = [];
    centerVector = maths.subtract(points[splitPoint - 1], points[splitPoint + 1]);
    if ((centerVector[0] === 0) && (centerVector[1] === 0)) {
        centerVector = maths.subtract(points[splitPoint - 1], points[splitPoint]).reverse();
        centerVector[0] = -centerVector[0];
    }
    const toCenterTangent = maths.normalize(centerVector);
    const fromCenterTangent = maths.mulItems(toCenterTangent, -1);
    console.log('before', beziers);
    beziers = beziers.concat(fitCubic(
        points.slice(0, splitPoint + 1), leftTangent,
        toCenterTangent, error, progressCallback,
    ));
    beziers = beziers.concat(fitCubic(
        points.slice(splitPoint), fromCenterTangent,
        rightTangent, error, progressCallback,
    ));
    console.log('after', beziers);
    return beziers;
}

function createTangent(pointA, pointB) {
    return maths.normalize(maths.subtract(pointA, pointB));
}

function fitCurve(points, maxError, progressCallback) {
    if (!Array.isArray(points)) {
        throw new TypeError('First argument should be an array');
    }
    points.forEach((point) => {
        if (!Array.isArray(point) || point.length !== 2
        || typeof point[0] !== 'number' || typeof point[1] !== 'number') {
            throw Error('Each point should be an array of two numbers');
        }
    });
    // Remove duplicate points
    const pointsFiltered = points.filter((point, i) =>
        i === 0 || !(point[0] === points[i - 1][0] && point[1] === points[i - 1][1]));

    if (pointsFiltered.length < 2) {
        return [];
    }

    const len = pointsFiltered.length;
    const leftTangent = createTangent(points[1], points[0]);
    const rightTangent = createTangent(points[len - 2], points[len - 1]);

    return fitCubic(points, leftTangent, rightTangent, maxError, progressCallback);
}

export default fitCurve;
